"""
Generate visualizations for the 3rd place March Mania 2026 solution.

Produces 4 figures saved to figures/ directory:
  1. LR coefficient importance (men's + women's)
  2. Brier score progression across 52 rounds
  3. Calibration plot (predicted probability vs actual win rate)
  4. Market blend impact (LR vs blended predictions)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from round27_pruned import _m_tourney, _w_tourney
from round51_final import build_m_features
from round50_final import build_w_features
from round52_final import (
    FEATURES_M,
    FEATURES_W,
    C_M,
    C_W,
    CLIP_M_LOW,
    CLIP_M_HIGH,
    CLIP_W_LOW,
    CLIP_W_HIGH,
    ALPHA_M_R1,
    ALPHA_M_BPI_BT,
    ALPHA_M_KALSHI,
    ALPHA_W_R1,
    ALPHA_W_BPI_BT,
    ALPHA_W_KALSHI,
)

HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
FIGURES.mkdir(exist_ok=True)
TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]

# Plot style
plt.rcParams.update(
    {
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#0e1117",
        "axes.edgecolor": "#333",
        "axes.labelcolor": "#eee",
        "text.color": "#eee",
        "xtick.color": "#ccc",
        "ytick.color": "#ccc",
        "grid.color": "#222",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 11,
    }
)
COLOR_M = "#ff6b35"  # orange for men's
COLOR_W = "#4ecdc4"  # teal for women's
COLOR_ACCENT = "#ffd166"  # gold


def _build_data(gender: str):
    """Build feature matrix and labels for a gender."""
    tourney = _m_tourney if gender == "M" else _w_tourney
    build_fn = build_m_features if gender == "M" else build_w_features
    feats = list(FEATURES_M) if gender == "M" else list(FEATURES_W)

    rows: list[dict] = []
    for season in TRAIN_SEASONS:
        games = tourney[tourney["Season"] == season]
        if len(games) == 0:
            continue
        features = build_fn(season)
        for _, g in games.iterrows():
            t1 = min(int(g["WTeamID"]), int(g["LTeamID"]))
            t2 = max(int(g["WTeamID"]), int(g["LTeamID"]))
            row: dict = {
                "Season": float(season),
                "Team1Win": 1.0 if int(g["WTeamID"]) == t1 else 0.0,
            }
            for feat in feats:
                v1 = features.loc[t1, feat] if t1 in features.index else np.nan
                v2 = features.loc[t2, feat] if t2 in features.index else np.nan
                row[f"d_{feat}"] = v1 - v2
            rows.append(row)

    df = pd.DataFrame(rows)
    feat_cols = [f"d_{f}" for f in feats]
    X = df[feat_cols].values.astype(np.float32)
    y = df["Team1Win"].values.astype(np.float32)
    groups = df["Season"].values.astype(int)
    return X, y, groups, feats


def _fit_pipeline(X, y, C):
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
        ]
    )
    pipe.fit(X, y)
    return pipe


def _get_oof(X, y, groups, C, clip_low, clip_high):
    """Get out-of-fold predictions via LOSO CV."""
    gkf = GroupKFold(n_splits=len(set(groups)))
    oof = np.zeros(len(y))
    for train_idx, val_idx in gkf.split(X, y, groups):
        pipe = _fit_pipeline(X[train_idx], y[train_idx], C)
        oof[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
    return np.clip(oof, clip_low, clip_high)


# ── Figure 1: LR Coefficient Importance ─────────────────────────


def plot_coefficients():
    print("  Generating coefficient importance plot...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    for ax, gender, color, title in [
        (ax1, "M", COLOR_M, "Men's Model (25 features, C=100)"),
        (ax2, "W", COLOR_W, "Women's Model (11 features, C=0.15)"),
    ]:
        feats = list(FEATURES_M) if gender == "M" else list(FEATURES_W)
        C = C_M if gender == "M" else C_W
        X, y, _, _ = _build_data(gender)
        pipe = _fit_pipeline(X, y, C)

        coefs = pipe.named_steps["lr"].coef_[0]
        # Scale by feature std to get "importance" (coef * std after imputation)
        scaler = pipe.named_steps["scaler"]
        scaled_coefs = coefs  # already on standardized scale

        sorted_idx = np.argsort(np.abs(scaled_coefs))
        sorted_feats = [feats[i] for i in sorted_idx]
        sorted_coefs = scaled_coefs[sorted_idx]

        colors = [color if c > 0 else "#e63946" for c in sorted_coefs]
        ax.barh(
            range(len(sorted_feats)), sorted_coefs, color=colors, alpha=0.85, height=0.7
        )
        ax.set_yticks(range(len(sorted_feats)))
        ax.set_yticklabels(sorted_feats, fontsize=9)
        ax.set_xlabel("Standardized Coefficient")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.axvline(0, color="#555", linewidth=0.8)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle(
        "Logistic Regression Feature Importance", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()
    fig.savefig(FIGURES / "01_coefficients.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved figures/01_coefficients.png")


# ── Figure 2: Brier Score Progression ────────────────────────────


def plot_progression():
    print("  Generating Brier score progression plot...")

    # Manually recorded progression from 52 rounds of iteration
    rounds = [
        ("R1\nXGB baseline", 0.158),
        ("R11\nXGB optimized", 0.157),
        ("R23\nSwitch to LR", 0.145),
        ("R24\nLR tuned", 0.142),
        ("R27\nFeature pruning\n33→18 M feats", 0.135),
        ("R34\nColley+GLM\n+interactions", 0.135),
        ("R46\nKalshi+SRS\n+C tuning", 0.128),
        ("R50\nESPN BPI\n+feature opt", 0.126),
        ("R51\nCoach PASE\n+W BPI", 0.125),
        ("R52\nTriple-market\nblend", 0.124),
    ]

    labels, scores = zip(*rounds)

    fig, ax = plt.subplots(figsize=(14, 6))

    # Line
    ax.plot(
        range(len(scores)),
        scores,
        color=COLOR_ACCENT,
        linewidth=2.5,
        marker="o",
        markersize=10,
        markeredgecolor="#0e1117",
        markeredgewidth=2,
        zorder=3,
    )

    # Fill below
    ax.fill_between(
        range(len(scores)), scores, min(scores) - 0.005, alpha=0.15, color=COLOR_ACCENT
    )

    # Annotations for key transitions
    transitions = {
        2: ("Switch to LR\n-0.013 Brier", COLOR_M),
        4: ("Pruning\n-0.007", COLOR_W),
        9: ("Market blend", "#e63946"),
    }
    for idx, (label, col) in transitions.items():
        ax.annotate(
            label,
            xy=(idx, scores[idx]),
            xytext=(idx, scores[idx] + 0.006),
            ha="center",
            fontsize=9,
            color=col,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=col, lw=1),
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8, ha="center")
    ax.set_ylabel("LOSO CV Brier Score (lower = better)")
    ax.set_title(
        "52 Rounds of Iteration: XGBoost → Logistic Regression → Market Blend",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(min(scores) - 0.005, max(scores) + 0.012)

    plt.tight_layout()
    fig.savefig(FIGURES / "02_progression.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved figures/02_progression.png")


# ── Figure 3: Calibration Plot ───────────────────────────────────


def plot_calibration():
    print("  Generating calibration plot...")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Perfect calibration line
    ax.plot(
        [0, 1], [0, 1], "--", color="#555", linewidth=1.5, label="Perfect calibration"
    )

    for gender, color, label in [
        ("M", COLOR_M, "Men's"),
        ("W", COLOR_W, "Women's"),
    ]:
        C = C_M if gender == "M" else C_W
        clip_lo = CLIP_M_LOW if gender == "M" else CLIP_W_LOW
        clip_hi = CLIP_M_HIGH if gender == "M" else CLIP_W_HIGH
        X, y, groups, _ = _build_data(gender)
        oof = _get_oof(X, y, groups, C, clip_lo, clip_hi)

        brier = brier_score_loss(y, oof)
        prob_true, prob_pred = calibration_curve(y, oof, n_bins=10, strategy="uniform")

        ax.plot(
            prob_pred,
            prob_true,
            "o-",
            color=color,
            linewidth=2,
            markersize=8,
            label=f"{label} (Brier={brier:.4f})",
        )

        # Show sample count per bin
        counts, edges = np.histogram(oof, bins=10, range=(0, 1))
        for i, (pp, pt) in enumerate(zip(prob_pred, prob_true)):
            if counts[i] > 10:
                ax.annotate(
                    f"n={counts[i]}",
                    xy=(pp, pt),
                    xytext=(5, -12),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                    alpha=0.7,
                )

    ax.set_xlabel("Predicted Probability (Team 1 wins)")
    ax.set_ylabel("Actual Win Rate")
    ax.set_title(
        "Calibration: Predicted vs Actual Win Probability (LOSO CV)",
        fontsize=13,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES / "03_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved figures/03_calibration.png")


# ── Figure 4: Market Blend Impact ────────────────────────────────


def plot_blend_impact():
    print("  Generating market blend impact diagram...")

    fig, ax = plt.subplots(figsize=(12, 7))

    # Tier data
    tiers = [
        {
            "label": "Tier 1: Round 1\nGame-Specific",
            "sources": "ESPN BPI + Vegas\nMoneylines",
            "m_weight": f"{ALPHA_M_R1:.0%} market",
            "w_weight": f"{ALPHA_W_R1:.0%} market",
            "coverage": "30 M + 28 W\nR1 games",
            "color": "#e63946",
        },
        {
            "label": "Tier 2: BPI\nBradley-Terry",
            "sources": "ESPN Championship\nOdds → pairwise",
            "m_weight": f"{ALPHA_M_BPI_BT:.0%} market",
            "w_weight": f"{ALPHA_W_BPI_BT:.0%} market",
            "coverage": "~64 M + ~64 W\nranked teams",
            "color": COLOR_M,
        },
        {
            "label": "Tier 3: Kalshi\nBradley-Terry",
            "sources": "Kalshi Futures\n→ pairwise",
            "m_weight": f"{ALPHA_M_KALSHI:.0%} market",
            "w_weight": f"{ALPHA_W_KALSHI:.0%} market",
            "coverage": "~40 M + ~25 W\nwith odds",
            "color": COLOR_W,
        },
        {
            "label": "Tier 4: Pure\nLogistic Regression",
            "sources": "25 M / 11 W\ndifferential features",
            "m_weight": "100% LR",
            "w_weight": "100% LR",
            "coverage": "All remaining\nmatchups",
            "color": COLOR_ACCENT,
        },
    ]

    y_positions = [3, 2, 1, 0]

    for i, (tier, ypos) in enumerate(zip(tiers, y_positions)):
        # Tier box
        rect = plt.Rectangle(
            (0.5, ypos - 0.35),
            2.5,
            0.7,
            facecolor=tier["color"],
            alpha=0.25,
            edgecolor=tier["color"],
            linewidth=2,
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(
            1.75,
            ypos,
            tier["label"],
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
            color=tier["color"],
            zorder=3,
        )

        # Source box
        rect2 = plt.Rectangle(
            (3.5, ypos - 0.35),
            2.5,
            0.7,
            facecolor="#1a1a2e",
            edgecolor="#333",
            linewidth=1,
            zorder=2,
        )
        ax.add_patch(rect2)
        ax.text(
            4.75,
            ypos,
            tier["sources"],
            ha="center",
            va="center",
            fontsize=9,
            color="#ccc",
            zorder=3,
        )

        # Arrow
        ax.annotate(
            "",
            xy=(3.45, ypos),
            xytext=(3.05, ypos),
            arrowprops=dict(arrowstyle="->", color=tier["color"], lw=2),
        )

        # Weight badges
        ax.text(
            6.5,
            ypos + 0.12,
            f"M: {tier['m_weight']}",
            ha="center",
            fontsize=9,
            color=COLOR_M,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#1a1a2e",
                edgecolor=COLOR_M,
                alpha=0.8,
            ),
        )
        ax.text(
            6.5,
            ypos - 0.22,
            f"W: {tier['w_weight']}",
            ha="center",
            fontsize=9,
            color=COLOR_W,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="#1a1a2e",
                edgecolor=COLOR_W,
                alpha=0.8,
            ),
        )

        # Coverage
        ax.text(
            8.2,
            ypos,
            tier["coverage"],
            ha="center",
            va="center",
            fontsize=8,
            color="#999",
        )

        # Fallback arrow to next tier
        if i < 3:
            ax.annotate(
                "fallback",
                xy=(1.75, ypos - 0.4),
                xytext=(1.75, ypos - 0.6),
                ha="center",
                fontsize=7,
                color="#666",
                arrowprops=dict(arrowstyle="->", color="#444", lw=1),
            )

    # Headers
    ax.text(
        1.75,
        3.7,
        "BLEND TIER",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#888",
    )
    ax.text(
        4.75,
        3.7,
        "DATA SOURCE",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#888",
    )
    ax.text(
        6.5,
        3.7,
        "MARKET WEIGHT",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="#888",
    )
    ax.text(
        8.2, 3.7, "COVERAGE", ha="center", fontsize=10, fontweight="bold", color="#888"
    )

    ax.set_xlim(0, 9.5)
    ax.set_ylim(-0.8, 4.2)
    ax.set_title(
        "Tiered Market Blend Architecture", fontsize=15, fontweight="bold", pad=15
    )
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(FIGURES / "04_blend_tiers.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("    Saved figures/04_blend_tiers.png")


if __name__ == "__main__":
    print("Generating visualizations...\n")
    plot_coefficients()
    plot_progression()
    plot_calibration()
    plot_blend_impact()
    print("\nDone! All figures saved to figures/")
