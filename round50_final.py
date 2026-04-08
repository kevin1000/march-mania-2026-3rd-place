"""
March Machine Learning Mania 2026 — Round 50: Single Best Submission

Strategy: CV-optimized LR + ESPN BPI market blend
  - M: LR + BPI (game-specific R1, Bradley-Terry for rest)
  - W: LR + Kalshi Bradley-Terry (no BPI for women's)

R50 CV improvements over R49 (0.12621):
  - M: drop pt_diff (redundant with SRS), +srs_x_ap_rank interaction
  - W: drop 4 features (last_n_pt_diff, colley_rank, elo, net_rank)
  - C retune: M 100→500, W 0.12→0.15
  - Clip retune: M [0.03,0.97], W [0.005,0.995]
  - R50 CV: M=0.11541, W=0.13506, Combined=0.12524 (Δ=-0.00097)

Blend alphas (market weight):
  - R1 known games: 0.50 (BPI has injuries, travel, matchup info)
  - Other M tournament matchups: 0.35 (BPI BT is weaker signal)
  - W tournament matchups: 0.15 (thin Kalshi market)
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
from round49_final import (
    build_m_features as _build_m_r49,
    build_w_features,
)
from round46_final import (
    load_kalshi_odds,
    kalshi_pairwise,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SUBMISSIONS = HERE / "submissions"

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]

# ── R50 optimized parameters (from round50_ideas.py experiments) ──

# M: 24 features (drop pt_diff, add srs_x_ap_rank)
FEATURES_M = [
    "SeedNum",
    "win_pct",
    "adjoe",
    "adjde",
    "WAB",
    "sos",
    "adjt",
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "oreb_pct_d",
    "fg3_pct",
    "ap_rank",
    "em_change",
    "glm_quality",
    "SeedNum_x_pt_diff",
    "massey_rank_x_barthag",
    "close_win_pct_x_pt_diff",
    "colley_rank",
    "ncsos",
    "SeedNum_x_massey",
    "ncsos_x_massey",
    "srs",
    "srs_x_ap_rank",
]

# W: 11 features (drop last_n_pt_diff, colley_rank, elo, net_rank)
FEATURES_W = [
    "SeedNum",
    "win_pct",
    "oreb_pct_d",
    "blk_pct",
    "last_n_win_pct",
    "elo_slope",
    "net_x_elo",
    "elo_x_colley",
    "srs",
    "last_n_pt_diff_x_elo_x_colley",
    "last_n_pt_diff_x_colley_rank",
]

C_M = 500.0  # R50: up from 100 (with srs_x_ap_rank, less regularization)
C_W = 0.15  # R50: up from 0.12
CLIP_M_LOW, CLIP_M_HIGH = 0.03, 0.97  # R50: tighter from [0.04, 0.96]
CLIP_W_LOW, CLIP_W_HIGH = 0.005, 0.995  # R50: wider from [0.01, 0.99]


def build_m_features(season: int) -> pd.DataFrame:
    """R49 M features + srs_x_ap_rank interaction."""
    features = _build_m_r49(season)
    if "srs" in features.columns and "ap_rank" in features.columns:
        features["srs_x_ap_rank"] = features["srs"] * features["ap_rank"]
    return features


# ── ESPN BPI data (scraped March 19, 2026) ───────────────────────

# Team name → TeamID mapping (extends KALSHI_M_MAP with tournament teams)
BPI_TEAM_MAP: dict[str, int] = {
    # From KALSHI_M_MAP (existing)
    "Duke": 1181,
    "Michigan": 1276,
    "Arizona": 1112,
    "Houston": 1222,
    "Florida": 1196,
    "Iowa State": 1235,
    "Purdue": 1345,
    "Illinois": 1228,
    "Gonzaga": 1211,
    "UConn": 1163,
    "Michigan State": 1277,
    "Tennessee": 1397,
    "Louisville": 1257,
    "Alabama": 1104,
    "Arkansas": 1116,
    "Vanderbilt": 1435,
    "Nebraska": 1304,
    "Virginia": 1438,
    "St. John's": 1385,
    "Kansas": 1242,
    "Texas Tech": 1403,
    "Kentucky": 1246,
    "BYU": 1140,
    "Wisconsin": 1458,
    "UCLA": 1417,
    "Ohio State": 1326,
    "North Carolina": 1314,
    "Clemson": 1155,
    "Saint Mary's": 1388,
    "Georgia": 1208,
    "Iowa": 1234,
    "Saint Louis": 1387,
    "Villanova": 1437,
    "Texas A&M": 1401,
    "Texas": 1400,
    "Miami": 1274,
    "Missouri": 1281,
    "High Point": 1219,
    "Radford": 1347,
    "UCF": 1416,
    "SMU": 1374,
    # New tournament teams (not in Kalshi)
    "Utah State": 1429,
    "McNeese": 1270,
    "Northern Iowa": 1320,
    "Miami (OH)": 1275,
    "Hofstra": 1220,
    "Hawaii": 1218,
    "Kennesaw State": 1244,
    "Troy": 1407,
    "North Dakota State": 1295,
    "California Baptist": 1465,
    "Wright State": 1460,
    "Queens": 1474,
    "Pennsylvania": 1335,
    "Siena": 1373,
    "Furman": 1202,
    "Howard": 1224,
    "Idaho": 1225,
    "Long Island University": 1254,
    "Tennessee State": 1398,
    "South Florida": 1378,
    "Prairie View A&M": 1341,
    "Lehigh": 1250,
    "Akron": 1103,
    "VCU": 1433,
}

# BPI championship probabilities (%) — higher = stronger team
BPI_M_CHAMP: dict[int, float] = {
    BPI_TEAM_MAP[name]: prob
    for name, prob in {
        "Duke": 24.4,
        "Michigan": 15.3,
        "Arizona": 14.0,
        "Houston": 9.4,
        "Florida": 7.7,
        "Iowa State": 5.9,
        "Purdue": 4.3,
        "Illinois": 4.2,
        "Gonzaga": 3.7,
        "UConn": 2.5,
        "Michigan State": 1.3,
        "Tennessee": 1.0,
        "Louisville": 0.9,
        "Alabama": 0.9,
        "Arkansas": 0.6,
        "Vanderbilt": 0.6,
        "Nebraska": 0.6,
        "Virginia": 0.5,
        "St. John's": 0.5,
        "Kansas": 0.3,
        "Texas Tech": 0.2,
        "Kentucky": 0.2,
        "BYU": 0.2,
        "Wisconsin": 0.1,
        "UCLA": 0.1,
        "Ohio State": 0.05,
        "North Carolina": 0.05,
        "Clemson": 0.05,
        "Saint Mary's": 0.05,
        "Georgia": 0.05,
        "Utah State": 0.05,
        "Iowa": 0.05,
        "Saint Louis": 0.03,
        "Villanova": 0.03,
        "Texas A&M": 0.03,
        "Texas": 0.03,
        "Miami": 0.02,
        "Missouri": 0.02,
        "McNeese": 0.01,
        "High Point": 0.01,
        "Akron": 0.01,
        "Northern Iowa": 0.01,
        "Miami (OH)": 0.01,
        "Hofstra": 0.01,
        "Hawaii": 0.01,
        "Kennesaw State": 0.01,
        "Troy": 0.01,
        "North Dakota State": 0.01,
        "California Baptist": 0.01,
        "Wright State": 0.01,
        "Queens": 0.01,
        "Pennsylvania": 0.01,
        "Siena": 0.01,
        "Furman": 0.01,
        "Howard": 0.01,
        "Idaho": 0.01,
        "Long Island University": 0.01,
        "Tennessee State": 0.01,
        "South Florida": 0.01,
        "Prairie View A&M": 0.01,
        "Lehigh": 0.01,
        "SMU": 0.05,
        "UCF": 0.02,
        "VCU": 0.02,
        "Radford": 0.01,
    }.items()
}

# BPI game-specific R1 predictions: {(lower_team_id, higher_team_id): P(lower wins)}
# ESPN BPI Mar 19 2026 — factors in injuries, travel, rest, matchup
_BPI_R1_RAW: list[tuple[str, str, float]] = [
    # (favored_team, underdog_team, favored_win_prob)
    ("Duke", "Siena", 0.990),
    ("Michigan", "Howard", 0.989),
    ("Arizona", "Long Island University", 0.990),
    ("Iowa State", "Tennessee State", 0.984),
    ("Houston", "Idaho", 0.982),
    ("Purdue", "Queens", 0.978),
    ("Illinois", "Pennsylvania", 0.973),
    ("UConn", "Furman", 0.964),
    ("Gonzaga", "Kennesaw State", 0.956),
    ("Michigan State", "North Dakota State", 0.942),
    ("Virginia", "Wright State", 0.930),
    ("Nebraska", "Troy", 0.927),
    ("Kansas", "California Baptist", 0.926),
    ("Arkansas", "Hawaii", 0.916),
    ("Alabama", "Hofstra", 0.901),
    ("St. John's", "Northern Iowa", 0.850),
    ("Vanderbilt", "McNeese", 0.812),
    ("Texas Tech", "Akron", 0.819),
    ("Louisville", "South Florida", 0.810),
    ("SMU", "Miami (OH)", 0.745),
    ("Wisconsin", "High Point", 0.763),
    ("UCLA", "UCF", 0.722),
    ("Kentucky", "Santa Clara", 0.722),
    ("Ohio State", "TCU", 0.654),
    ("BYU", "Texas", 0.630),
    ("North Carolina", "VCU", 0.616),
    ("Miami", "Missouri", 0.551),
    ("Saint Mary's", "Texas A&M", 0.532),
    ("Utah State", "Villanova", 0.517),
    ("Lehigh", "Prairie View A&M", 0.537),
    ("Georgia", "Saint Louis", 0.509),
    ("Clemson", "Iowa", 0.506),
]


def _build_bpi_r1_games() -> dict[tuple[int, int], float]:
    """Convert BPI R1 predictions to {(min_id, max_id): P(min_id wins)} format."""
    # Need additional team mappings for teams only in R1 matchups
    extra_map = {
        "Santa Clara": 1365,
        "TCU": 1395,
    }
    full_map = {**BPI_TEAM_MAP, **extra_map}

    games: dict[tuple[int, int], float] = {}
    for fav, dog, fav_prob in _BPI_R1_RAW:
        fav_id = full_map.get(fav)
        dog_id = full_map.get(dog)
        if fav_id is None or dog_id is None:
            print(f"  WARNING: unmapped R1 team: {fav} or {dog}")
            continue
        t1 = min(fav_id, dog_id)
        t2 = max(fav_id, dog_id)
        # P(t1 wins) = P(fav wins) if t1==fav, else 1-P(fav wins)
        if t1 == fav_id:
            games[(t1, t2)] = fav_prob
        else:
            games[(t1, t2)] = 1.0 - fav_prob
    return games


BPI_R1_GAMES = _build_bpi_r1_games()


def bpi_bt_pairwise(t1: int, t2: int, floor: float = 0.005) -> float:
    """Bradley-Terry from BPI championship probabilities."""
    s1 = BPI_M_CHAMP.get(t1, floor)
    s2 = BPI_M_CHAMP.get(t2, floor)
    return s1 / (s1 + s2)


# ── CV evaluation ──────────────────────────────────────────────────


def evaluate_cv() -> None:
    """R50 CV evaluation with optimized features/params."""
    print("\n── CV Evaluation (R50 optimized) ──")

    briers: dict[str, float] = {}
    for gender, feats, C_val, clip_low, clip_high in [
        ("M", FEATURES_M, C_M, CLIP_M_LOW, CLIP_M_HIGH),
        ("W", FEATURES_W, C_W, CLIP_W_LOW, CLIP_W_HIGH),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney
        rows: list[dict[str, float]] = []
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


# ── Generate submission ────────────────────────────────────────────


def generate_submission(
    tag: str = "r50_best",
    m_alpha_r1: float = 0.50,
    m_alpha_bt: float = 0.35,
    w_alpha: float = 0.15,
) -> Path:
    """Generate single best submission with market blend.

    Args:
        m_alpha_r1: BPI weight for known M first-round games
        m_alpha_bt: BPI Bradley-Terry weight for other M matchups
        w_alpha: Kalshi weight for W matchups
    """
    m_odds, w_odds = load_kalshi_odds()
    print(f"  Kalshi: {len(m_odds)} M, {len(w_odds)} W teams")
    print(f"  BPI: {len(BPI_M_CHAMP)} M teams, {len(BPI_R1_GAMES)} R1 games")
    print(f"  Blend alphas: M_R1={m_alpha_r1}, M_BT={m_alpha_bt}, W={w_alpha}")

    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds: dict[str, float] = {}
    blend_stats = {"pure_lr": 0, "bpi_r1": 0, "bpi_bt": 0, "kalshi": 0}

    for gender, feats, C_val, clip_low, clip_high in [
        ("M", FEATURES_M, C_M, CLIP_M_LOW, CLIP_M_HIGH),
        ("W", FEATURES_W, C_W, CLIP_W_LOW, CLIP_W_HIGH),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney

        # Build training data
        rows: list[dict[str, float]] = []
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
                    "Team1Win": 1.0 if int(g["WTeamID"]) == t1 else 0.0,
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

        for _, row_sub in sub.iterrows():
            game_id = row_sub["ID"]
            parts = game_id.split("_")
            if len(parts) != 3:
                continue
            if int(parts[0]) != 2026:
                continue
            t1, t2 = int(parts[1]), int(parts[2])
            is_mens = t1 < 3000
            if (gender == "M" and not is_mens) or (gender == "W" and is_mens):
                continue

            # LR prediction
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
            lr_pred = float(pipe.predict_proba(x_arr)[:, 1][0])

            # Market blend
            if gender == "M":
                key = (t1, t2)
                if key in BPI_R1_GAMES:
                    # Known first-round game — use game-specific BPI
                    market_pred = BPI_R1_GAMES[key]
                    final = (1 - m_alpha_r1) * lr_pred + m_alpha_r1 * market_pred
                    blend_stats["bpi_r1"] += 1
                elif t1 in BPI_M_CHAMP and t2 in BPI_M_CHAMP:
                    # Both in BPI — use Bradley-Terry from championship probs
                    market_pred = bpi_bt_pairwise(t1, t2)
                    final = (1 - m_alpha_bt) * lr_pred + m_alpha_bt * market_pred
                    blend_stats["bpi_bt"] += 1
                elif t1 in m_odds and t2 in m_odds:
                    # Kalshi fallback
                    market_pred = kalshi_pairwise(t1, t2, m_odds)
                    final = 0.80 * lr_pred + 0.20 * market_pred
                    blend_stats["kalshi"] += 1
                else:
                    final = lr_pred
                    blend_stats["pure_lr"] += 1
            else:
                # Women's
                if t1 in w_odds and t2 in w_odds:
                    market_pred = kalshi_pairwise(t1, t2, w_odds)
                    final = (1 - w_alpha) * lr_pred + w_alpha * market_pred
                    blend_stats["kalshi"] += 1
                else:
                    final = lr_pred
                    blend_stats["pure_lr"] += 1

            preds[game_id] = float(np.clip(final, clip_low, clip_high))

    sub["Pred"] = sub["ID"].map(preds).fillna(0.5)
    SUBMISSIONS.mkdir(exist_ok=True)
    out_path = SUBMISSIONS / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\nSubmission saved: {out_path}")
    print(f"  Total predictions: {len(preds)}")
    print(f"  Pred range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
    print(f"  Mean pred: {sub['Pred'].mean():.4f}")
    print(f"  Blend stats: {blend_stats}")
    return out_path


# ── Sanity checks ─────────────────────────────────────────────────


def sanity_check() -> None:
    """Verify BPI blend gives reasonable predictions."""
    print("\n── Sanity Checks ──")

    # Check R1 game predictions
    print("\n  BPI R1 game-specific predictions (sample):")
    for (t1, t2), prob in sorted(BPI_R1_GAMES.items(), key=lambda x: -x[1])[:5]:
        print(f"    {t1} vs {t2}: P(t1 wins) = {prob:.3f}")
    for (t1, t2), prob in sorted(BPI_R1_GAMES.items(), key=lambda x: abs(x[1] - 0.5))[
        :5
    ]:
        print(f"    {t1} vs {t2}: P(t1 wins) = {prob:.3f} (close game)")

    # Check BT pairwise
    print("\n  BPI Bradley-Terry (sample):")
    # Duke (1181) vs Kentucky (1246)
    print(f"    Duke vs Kentucky: {bpi_bt_pairwise(1181, 1246):.3f} (expect ~0.99)")
    # Duke (1181) vs Michigan (1276)
    print(f"    Duke vs Michigan: {bpi_bt_pairwise(1181, 1276):.3f} (expect ~0.61)")
    # Ohio State (1326) vs Georgia (1208)
    print(f"    Ohio St vs Georgia: {bpi_bt_pairwise(1326, 1208):.3f} (expect ~0.50)")


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 70)
    print("March Mania 2026 — Round 50: Single Best Submission")
    print("  LR (R50 optimized) + ESPN BPI market blend")
    print(f"  M: {len(FEATURES_M)} features, C={C_M}")
    print(f"  W: {len(FEATURES_W)} features, C={C_W}")
    print(f"  BPI R1 games: {len(BPI_R1_GAMES)}")
    print(f"  BPI teams: {len(BPI_M_CHAMP)}")
    print("=" * 70)

    evaluate_cv()
    sanity_check()

    print("\n── Generating submissions ──")

    # 1. Pure R50 LR (no market blend) — for comparison
    print("\n1. Pure R50 LR (no market blend)")
    generate_submission(
        tag="r50_pure",
        m_alpha_r1=0.0,
        m_alpha_bt=0.0,
        w_alpha=0.0,
    )

    # 2. R50 LR + BPI/Kalshi blend (recommended)
    print("\n2. R50 LR + BPI/Kalshi market blend")
    generate_submission(
        tag="r50_best",
        m_alpha_r1=0.50,
        m_alpha_bt=0.35,
        w_alpha=0.15,
    )

    print("\n" + "=" * 70)
    print("DONE — submissions generated:")
    print("  1. submission_r50_pure.csv    — Pure CV-optimized LR")
    print("  2. submission_r50_best.csv    — LR + BPI/Kalshi market blend")
    print("=" * 70)
