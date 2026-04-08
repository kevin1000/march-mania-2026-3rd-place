"""
March Machine Learning Mania 2026 — Round 49: SRS + W interactions

R46 CV: 0.12777 (M=0.11914, W=0.13640)
R49 improvements:
  M: +SRS feature, C=750→100 → 0.11693 (Δ=-0.00221)
  W: +SRS +last_n_pt_diff_x_elo_x_colley +last_n_pt_diff_x_colley_rank, C=0.25→0.12
     → 0.13548 (Δ=-0.00092)
  Combined: 0.12621 (Δ=-0.00156)

SRS = Simple Rating System: iterative solution where
  rating_i = avg_margin_i + avg(rating_opponents_i)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from round27_pruned import (
    _m_tourney,
    _w_tourney,
)
from round46_final import (
    build_m_features as _build_m_r46,
    build_w_features as _build_w_r46,
    FEATURES_M as _FEATURES_M_R46,
    FEATURES_W as _FEATURES_W_R46,
    CLIP_M_LOW,
    CLIP_M_HIGH,
    CLIP_W_LOW,
    CLIP_W_HIGH,
    load_kalshi_odds,
    kalshi_pairwise,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SUBMISSIONS = HERE / "submissions"

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]

# R49 tuned parameters
C_M = 100.0  # R49: down from 750 (with SRS, more regularization helps)
C_W = 0.12  # R49: down from 0.25 (with new features, more regularization helps)

# Features
FEATURES_M = list(_FEATURES_M_R46) + ["srs"]
FEATURES_W = list(_FEATURES_W_R46) + [
    "srs",
    "last_n_pt_diff_x_elo_x_colley",
    "last_n_pt_diff_x_colley_rank",
]


# ── SRS computation ────────────────────────────────────────────────


def compute_srs(gender: str, season: int, max_iter: int = 100) -> dict[int, float]:
    """Simple Rating System: rating = avg_margin + avg(opp_ratings), iterative."""
    prefix = "M" if gender == "M" else "W"
    compact = pd.read_csv(DATA / f"{prefix}RegularSeasonCompactResults.csv")
    games = compact[compact["Season"] == season]

    teams = sorted(set(games["WTeamID"].tolist() + games["LTeamID"].tolist()))
    tid_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    ratings = np.zeros(n)
    margins = np.zeros(n)
    counts = np.zeros(n)
    opp_lists: list[list[int]] = [[] for _ in range(n)]

    for _, g in games.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        wi, li = tid_to_idx[w], tid_to_idx[l]
        margin = int(g["WScore"]) - int(g["LScore"])
        margins[wi] += margin
        margins[li] -= margin
        counts[wi] += 1
        counts[li] += 1
        opp_lists[wi].append(li)
        opp_lists[li].append(wi)

    avg_margins = margins / np.maximum(counts, 1)

    for _ in range(max_iter):
        new_ratings = np.zeros(n)
        for i in range(n):
            avg_opp = np.mean(ratings[opp_lists[i]]) if opp_lists[i] else 0.0
            new_ratings[i] = avg_margins[i] + avg_opp
        new_ratings -= new_ratings.mean()
        if np.max(np.abs(new_ratings - ratings)) < 1e-6:
            break
        ratings = new_ratings

    return {teams[i]: float(ratings[i]) for i in range(n)}


# Cache SRS computations
_srs_m_cache: dict[int, dict[int, float]] = {}
_srs_w_cache: dict[int, dict[int, float]] = {}


def build_m_features(season: int) -> pd.DataFrame:
    """R46 M features + SRS."""
    if season not in _srs_m_cache:
        _srs_m_cache[season] = compute_srs("M", season)
    features = _build_m_r46(season)
    srs = _srs_m_cache[season]
    features["srs"] = features.index.map(lambda t: srs.get(t, np.nan))
    return features


def build_w_features(season: int) -> pd.DataFrame:
    """R46 W features + SRS + interactions."""
    if season not in _srs_w_cache:
        _srs_w_cache[season] = compute_srs("W", season)
    features = _build_w_r46(season)
    srs = _srs_w_cache[season]
    features["srs"] = features.index.map(lambda t: srs.get(t, np.nan))

    # New interactions
    if "last_n_pt_diff" in features.columns and "elo_x_colley" in features.columns:
        features["last_n_pt_diff_x_elo_x_colley"] = (
            features["last_n_pt_diff"] * features["elo_x_colley"]
        )
    if "last_n_pt_diff" in features.columns and "colley_rank" in features.columns:
        features["last_n_pt_diff_x_colley_rank"] = (
            features["last_n_pt_diff"] * features["colley_rank"]
        )
    return features


# ── CV evaluation ──────────────────────────────────────────────────


def evaluate_cv() -> None:
    print("\n── CV Evaluation ──")

    briers = {}
    for gender, feats, C_val, clip_low, clip_high in [
        ("M", FEATURES_M, C_M, CLIP_M_LOW, CLIP_M_HIGH),
        ("W", FEATURES_W, C_W, CLIP_W_LOW, CLIP_W_HIGH),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney
        rows = []
        for season in TRAIN_SEASONS:
            games = tourney[tourney["Season"] == season]
            if len(games) == 0:
                continue
            features = (
                build_m_features(season) if gender == "M" else build_w_features(season)
            )
            for _, g in games.iterrows():
                t1 = min(int(g["WTeamID"]), int(g["LTeamID"]))
                t2 = max(int(g["WTeamID"]), int(g["LTeamID"]))
                row: dict[str, float] = {
                    "Season": season,
                    "Team1Win": 1 if int(g["WTeamID"]) == t1 else 0,
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

        gkf = GroupKFold(n_splits=len(set(groups)))
        oof = np.zeros(len(y))
        for train_idx, val_idx in gkf.split(X, y, groups):
            pipe = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=C_val, max_iter=1000, solver="lbfgs")),
                ]
            )
            pipe.fit(X[train_idx], y[train_idx])
            oof[val_idx] = np.clip(
                pipe.predict_proba(X[val_idx])[:, 1], clip_low, clip_high
            )
        brier = brier_score_loss(y, oof)
        briers[gender] = brier
        print(f"  {gender}: {brier:.5f} ({len(y)} games)")

    combined = 0.5 * (briers["M"] + briers["W"])
    print(f"  Combined: {combined:.5f}")
    print("  (R46 was: M=0.11914, W=0.13640, Combined=0.12777)")


# ── Generate submission ────────────────────────────────────────────


def generate_submission(
    tag: str = "r49_pure",
    kalshi_alpha: float = 0.0,
) -> Path:
    """Generate submission, optionally blending with Kalshi odds."""
    m_odds, w_odds = {}, {}
    if kalshi_alpha > 0:
        m_odds, w_odds = load_kalshi_odds()
        print(f"  Kalshi loaded: {len(m_odds)} M, {len(w_odds)} W teams")

    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds: dict[str, float] = {}

    for gender, feats, C_val, clip_low, clip_high, odds in [
        ("M", FEATURES_M, C_M, CLIP_M_LOW, CLIP_M_HIGH, m_odds),
        ("W", FEATURES_W, C_W, CLIP_W_LOW, CLIP_W_HIGH, w_odds),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney

        rows = []
        for season in TRAIN_SEASONS:
            games = tourney[tourney["Season"] == season]
            if len(games) == 0:
                continue
            features = (
                build_m_features(season) if gender == "M" else build_w_features(season)
            )
            for _, g in games.iterrows():
                t1 = min(int(g["WTeamID"]), int(g["LTeamID"]))
                t2 = max(int(g["WTeamID"]), int(g["LTeamID"]))
                row: dict[str, float] = {
                    "Team1Win": 1 if int(g["WTeamID"]) == t1 else 0,
                }
                for feat in feats:
                    v1 = features.loc[t1, feat] if t1 in features.index else np.nan
                    v2 = features.loc[t2, feat] if t2 in features.index else np.nan
                    row[f"d_{feat}"] = v1 - v2
                rows.append(row)

        df = pd.DataFrame(rows)
        feat_cols = [f"d_{f}" for f in feats]
        X_train = df[feat_cols].values.astype(np.float32)
        y_train = df["Team1Win"].values.astype(np.float32)

        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C_val, max_iter=1000, solver="lbfgs")),
            ]
        )
        pipe.fit(X_train, y_train)

        # Predict 2026
        features_2026 = (
            build_m_features(2026) if gender == "M" else build_w_features(2026)
        )

        for _, row in sub.iterrows():
            game_id = row["ID"]
            parts = game_id.split("_")
            if len(parts) != 3:
                continue
            season_str, t1_str, t2_str = parts
            if int(season_str) != 2026:
                continue
            t1, t2 = int(t1_str), int(t2_str)
            is_mens = t1 < 3000
            if (gender == "M" and not is_mens) or (gender == "W" and is_mens):
                continue

            x_row = []
            for feat in feats:
                v1 = (
                    features_2026.loc[t1, feat] if t1 in features_2026.index else np.nan
                )
                v2 = (
                    features_2026.loc[t2, feat] if t2 in features_2026.index else np.nan
                )
                x_row.append(v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)

            x_arr = np.array([x_row], dtype=np.float32)
            lr_pred = float(
                np.clip(pipe.predict_proba(x_arr)[:, 1][0], clip_low, clip_high)
            )

            if kalshi_alpha > 0 and (t1 in odds or t2 in odds):
                k_pred = kalshi_pairwise(t1, t2, odds)
                k_pred = float(np.clip(k_pred, 0.01, 0.99))
                final_pred = (1 - kalshi_alpha) * lr_pred + kalshi_alpha * k_pred
            else:
                final_pred = lr_pred

            preds[game_id] = float(np.clip(final_pred, clip_low, clip_high))

    sub["Pred"] = sub["ID"].map(preds).fillna(0.5)
    SUBMISSIONS.mkdir(exist_ok=True)
    out_path = SUBMISSIONS / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path}")
    print(f"  Total predictions: {len(preds)}")
    print(f"  Pred range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
    print(f"  Mean pred: {sub['Pred'].mean():.4f}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 70)
    print("March Mania 2026 — Round 49: SRS + W interactions")
    print(f"  M: {len(FEATURES_M)} features, C={C_M}")
    print(f"  W: {len(FEATURES_W)} features, C={C_W}")
    print("=" * 70)

    evaluate_cv()

    print("\n── Generating submissions ──")

    # Pure model
    print("\n1. Pure R49")
    generate_submission(tag="r49_pure", kalshi_alpha=0.0)

    # Kalshi 5% blend
    print("\n2. Kalshi 5% blend")
    generate_submission(tag="r49_kalshi_a05", kalshi_alpha=0.05)

    print("\n" + "=" * 70)
    print("DONE — R49: +SRS(M+W) +W_interactions, C retune")
    print("=" * 70)
