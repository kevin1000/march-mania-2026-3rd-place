"""
March Machine Learning Mania 2026 — Round 46: Final submission + Kalshi blend

R45 CV: 0.12797 (M=0.11913, W=0.13680)
R46 improvements:
  - W: +elo_x_colley interaction (Δ=-0.00019 on W)
  - W: C=0.15 → C=0.25 (jointly tuned with elo_x_colley)
  - M: C=300 → C=750 (Δ=-0.00004, marginal)
  - Expected: M=0.11914, W=0.13640, Combined=0.12777 (Δ=-0.00020)

Submissions:
  1. r46_pure — Pure R46 LR model
  2. r46_kalshi_a05 — 5% Kalshi blend
  3. r46_kalshi_a10 — 10% Kalshi blend
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
from round45_final import (
    build_m_features,
    build_w_features as _build_w_r45,
    FEATURES_M,
    FEATURES_W as _FEATURES_W_R45,
    CLIP_M_LOW,
    CLIP_M_HIGH,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXT = HERE / "external"
SUBMISSIONS = HERE / "submissions"

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]

# R46 tuned parameters
C_M = 750.0  # R46: up from 300 (Δ=-0.00004)
C_W = 0.25  # R46: jointly tuned with elo_x_colley (was 0.15)
CLIP_W_LOW, CLIP_W_HIGH = 0.01, 0.99

# W features: R45 + elo_x_colley interaction
FEATURES_W = list(_FEATURES_W_R45) + ["elo_x_colley"]


def build_w_features(season: int) -> pd.DataFrame:
    """R45 W features + elo_x_colley interaction."""
    features = _build_w_r45(season)
    if "elo" in features.columns and "colley_rank" in features.columns:
        features["elo_x_colley"] = features["elo"] * features["colley_rank"]
    return features


# Kalshi team name → TeamID mapping (from R30)
KALSHI_M_MAP: dict[str, int] = {
    "Duke": 1181,
    "Michigan": 1276,
    "Arizona": 1112,
    "Florida": 1196,
    "Houston": 1222,
    "UConn": 1163,
    "Illinois": 1228,
    "Iowa St.": 1235,
    "Purdue": 1345,
    "Michigan St.": 1277,
    "Vanderbilt": 1435,
    "Arkansas": 1116,
    "Wisconsin": 1458,
    "Gonzaga": 1211,
    "BYU": 1140,
    "St. John's": 1385,
    "Kansas": 1242,
    "Tennessee": 1397,
    "Mississippi State": 1280,
    "Indiana": 1231,
    "Ole Miss": 1279,
    "Marquette": 1266,
    "Memphis": 1272,
    "Stanford": 1390,
    "San Francisco": 1362,
    "Cincinnati": 1153,
    "Xavier": 1462,
    "Kentucky": 1246,
    "Baylor": 1124,
    "Virginia Tech": 1439,
    "Missouri": 1281,
    "Oregon": 1332,
    "Villanova": 1437,
    "Texas Tech": 1403,
    "Oklahoma": 1328,
    "North Carolina": 1314,
    "Seton Hall": 1371,
    "Georgia": 1208,
    "Maryland": 1268,
    "Ohio State": 1326,
    "USC": 1425,
    "Boston College": 1130,
    "Creighton": 1166,
    "UCF": 1416,
    "Boise St.": 1129,
    "VCU": 1433,
    "Nebraska": 1304,
    "New Mexico": 1307,
    "Colorado": 1160,
    "Auburn": 1120,
    "Clemson": 1155,
    "Alabama": 1104,
    "Drake": 1179,
    "Texas": 1400,
    "Pittsburgh": 1338,
    "High Point": 1219,
    "Radford": 1347,
    "Saint Louis": 1387,
    "UC San Diego": 1471,
    "Presbyterian": 1342,
    "Loyola Chicago": 1260,
    "Santa Clara": 1363,
    "Colorado St.": 1159,
    "Louisville": 1257,
    "Florida St.": 1197,
    "St. Bonaventure": 1386,
    "Syracuse": 1393,
    "Miami (FL)": 1274,
    "Notre Dame": 1323,
    "Longwood": 1263,
    "Nevada": 1305,
    "San Diego St.": 1361,
    "Dayton": 1172,
    "North Carolina St.": 1316,
    "Richmond": 1351,
    "Texas A&M": 1401,
    "Saint Mary's": 1388,
    "Georgetown": 1210,
    "UCLA": 1417,
    "Iowa": 1234,
    "SMU": 1380,
    "Wake Forest": 1441,
    "Butler": 1143,
    "Providence": 1344,
    "Virginia": 1438,
    "USC Upstate": 1426,
}
KALSHI_W_MAP: dict[str, int] = {
    "UConn": 3163,
    "UCLA": 3417,
    "South Carolina": 3381,
    "Texas": 3400,
    "LSU": 3261,
    "North Carolina": 3314,
    "Notre Dame": 3323,
    "Iowa St.": 3235,
    "Iowa": 3234,
    "Duke": 3181,
    "Baylor": 3124,
    "North Carolina St.": 3316,
    "Kentucky": 3246,
    "Louisville": 3258,
    "USC": 3425,
    "TCU": 3394,
    "Tennessee": 3397,
    "Oklahoma": 3328,
    "Vanderbilt": 3435,
    "Michigan": 3276,
    "Ole Miss": 3279,
    "Rhode Island": 3348,
}


def load_kalshi_odds() -> tuple[dict[int, float], dict[int, float]]:
    """Load Kalshi championship odds → {TeamID: implied_prob}."""
    m_odds: dict[int, float] = {}
    w_odds: dict[int, float] = {}

    kalshi_m = pd.read_csv(EXT / "kalshi_ncaam_champ_2026.csv")
    for _, row in kalshi_m.iterrows():
        team_name = row["team"]
        prob = row["mid_prob"]
        if team_name in KALSHI_M_MAP and prob > 0:
            m_odds[KALSHI_M_MAP[team_name]] = prob

    kalshi_w = pd.read_csv(EXT / "kalshi_ncaaw_champ_2026.csv")
    for _, row in kalshi_w.iterrows():
        team_name = row["team"]
        prob = row["mid_prob"]
        if team_name in KALSHI_W_MAP and prob > 0:
            w_odds[KALSHI_W_MAP[team_name]] = prob

    return m_odds, w_odds


def kalshi_pairwise(
    t1: int, t2: int, odds: dict[int, float], floor: float = 0.005
) -> float:
    """Bradley-Terry: P(t1 beats t2) = strength_t1 / (strength_t1 + strength_t2)."""
    s1 = odds.get(t1, floor)
    s2 = odds.get(t2, floor)
    return s1 / (s1 + s2)


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
    print("  (R45 was: M=0.11913, W=0.13680, Combined=0.12797)")


# ── Generate submission ────────────────────────────────────────────


def generate_submission(
    tag: str = "r46_pure",
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
        n_kalshi = 0

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
                if t1 in odds and t2 in odds:
                    n_kalshi += 1
            else:
                final_pred = lr_pred

            preds[game_id] = float(np.clip(final_pred, clip_low, clip_high))

        if kalshi_alpha > 0:
            print(f"  {gender}: {n_kalshi} matchups with both teams in Kalshi")

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
    print("March Mania 2026 — Round 46: Final submission")
    print("  M: 23 features, C=750 (R45 features)")
    print("  W: 12 features, C=0.25 (R46: +elo_x_colley, C tuned)")
    print("=" * 70)

    evaluate_cv()

    print("\n── Generating submissions ──")

    # Pure model
    print("\n1. Pure R46 (no Kalshi blend)")
    generate_submission(tag="r46_pure", kalshi_alpha=0.0)

    # Kalshi blends
    for alpha in [0.05, 0.10]:
        print(f"\n2. Kalshi blend α={alpha}")
        generate_submission(
            tag=f"r46_kalshi_a{int(alpha * 100):02d}", kalshi_alpha=alpha
        )

    print("\n" + "=" * 70)
    print("DONE — Submit r46_pure + r46_kalshi_a05 as alternatives")
    print("=" * 70)
