"""
March Machine Learning Mania 2026 — Round 45: Drop massey_rank (M)

R45: massey_rank standalone redundant with SeedNum_x_massey + ncsos_x_massey interactions.
M: 23 features (R44 24 - massey_rank), C=300
W: 11 features (unchanged from R44), C=0.15
Baseline: R44 CV 0.12817 (M=0.11953, W=0.13680)
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
    _m_reg,
    _w_reg,
    _m_tourney,
    _w_tourney,
    _massey,
    build_team_features,
    expected_score,
    mov_multiplier,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXT = HERE / "external"
SUBMISSIONS = HERE / "submissions"

_bt_map = pd.read_csv(EXT / "barttorvik_team_id_map.csv")

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]

CLIP_M_LOW, CLIP_M_HIGH = 0.04, 0.96  # R42: [0.04, 0.96] marginal improvement
CLIP_W_LOW, CLIP_W_HIGH = 0.01, 0.99  # Keep R40 values (R42 changes hurt in eval_cv)

# M features: R40 pruned — drop barthag, close_win_pct, last_n_pt_diff
FEATURES_M = [
    "SeedNum",
    "win_pct",
    "pt_diff",
    "adjoe",
    "adjde",
    # barthag: REMOVED R42 (Δ=-0.00034)
    "WAB",
    "sos",
    "adjt",
    # massey_rank: REMOVED R45 (Δ=-0.00037, redundant with interactions)
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "oreb_pct_d",
    "fg3_pct",
    # last_n_pt_diff: REMOVED R42 (Δ=-0.00026)
    "ap_rank",
    "em_change",
    # close_win_pct: REMOVED R42 (Δ=-0.00027)
    "glm_quality",
    "SeedNum_x_pt_diff",
    "massey_rank_x_barthag",  # interaction still valuable (+0.00163)
    "close_win_pct_x_pt_diff",  # interaction still valuable (+0.00059)
    "colley_rank",
    "ncsos",
    "SeedNum_x_massey",  # R43: interaction (Δ=-0.00073)
    "ncsos_x_massey",  # R44: interaction (Δ=-0.00047)
]
# W features: R44 adds net_x_elo interaction
FEATURES_W = [
    "SeedNum",
    "win_pct",
    "elo",
    "oreb_pct_d",
    "blk_pct",
    "last_n_win_pct",
    "last_n_pt_diff",
    "net_rank",
    "elo_slope",
    "colley_rank",
    "net_x_elo",  # R44: interaction (Δ=-0.00034)
]
C_M, C_W = 300.0, 0.15  # R44: W C=0.15 (Δ=-0.00034 with net_x_elo)


# ── Close game record ─────────────────────────────────────────────


def _compute_close_game_record(
    season: int,
    margin: int = 7,
) -> pd.DataFrame:
    reg = _m_reg[_m_reg["Season"] == season]
    close = reg[(reg["WScore"] - reg["LScore"]) <= margin]
    if len(close) == 0:
        return pd.DataFrame(columns=["TeamID", "close_win_pct"]).set_index("TeamID")
    w_rows = pd.DataFrame({"TeamID": close["WTeamID"], "Win": 1})
    l_rows = pd.DataFrame({"TeamID": close["LTeamID"], "Win": 0})
    all_games = pd.concat([w_rows, l_rows], ignore_index=True)
    stats = (
        all_games.groupby("TeamID")["Win"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "close_win_pct", "count": "close_games"})
    )
    stats.loc[stats["close_games"] < 3, "close_win_pct"] = np.nan
    return stats[["close_win_pct"]]


# ── Accuracy-weighted Massey composite ─────────────────────────────


_massey_weights_cache: dict[str, float] | None = None
_massey_full_coverage_cache: list[str] | None = None


def _get_massey_system_weights() -> tuple[dict[str, float], list[str]]:
    """Compute and cache Massey system weights (same for all seasons)."""
    global _massey_weights_cache, _massey_full_coverage_cache
    if _massey_weights_cache is not None:
        return _massey_weights_cache, _massey_full_coverage_cache  # type: ignore[return-value]

    available = _massey[_massey["Season"].isin(TRAIN_SEASONS)]
    system_coverage = available.groupby("SystemName")["Season"].nunique()
    full_coverage = system_coverage[
        system_coverage == len(TRAIN_SEASONS)
    ].index.tolist()

    system_weights: dict[str, float] = {}
    for system in full_coverage:
        correct = 0
        total = 0
        for eval_season in TRAIN_SEASONS:
            sys_data = _massey[
                (_massey["Season"] == eval_season) & (_massey["SystemName"] == system)
            ]
            if len(sys_data) == 0:
                continue
            latest = sys_data.loc[sys_data.groupby("TeamID")["RankingDayNum"].idxmax()]
            ranks = latest.set_index("TeamID")["OrdinalRank"]
            games = _m_tourney[_m_tourney["Season"] == eval_season]
            for _, g in games.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                if w in ranks.index and l in ranks.index:
                    if ranks[w] < ranks[l]:
                        correct += 1
                    total += 1
        if total > 0:
            system_weights[system] = correct / total

    if not system_weights:
        _massey_weights_cache = {}
        _massey_full_coverage_cache = []
        return {}, []

    weights = {s: max(acc - 0.5, 0.01) for s, acc in system_weights.items()}
    total_w = sum(weights.values())
    weights = {s: w / total_w for s, w in weights.items()}

    _massey_weights_cache = weights
    _massey_full_coverage_cache = full_coverage
    return weights, full_coverage


def _get_massey_weighted(season: int) -> pd.DataFrame:
    """Weight Massey systems by historical tournament predictive accuracy."""
    weights, full_coverage = _get_massey_system_weights()

    if not weights:
        return pd.DataFrame(columns=["TeamID", "massey_rank"]).set_index("TeamID")

    m = _massey[
        (_massey["Season"] == season) & (_massey["SystemName"].isin(full_coverage))
    ]
    if len(m) == 0:
        return pd.DataFrame(columns=["TeamID", "massey_rank"]).set_index("TeamID")

    latest = m.groupby(["SystemName", "TeamID"])["RankingDayNum"].max().reset_index()
    latest = latest.merge(m, on=["SystemName", "TeamID", "RankingDayNum"])

    # Weighted average
    team_scores: dict[int, float] = {}
    team_weight_sums: dict[int, float] = {}
    for _, row in latest.iterrows():
        sys_name = row["SystemName"]
        if sys_name not in weights:
            continue
        team = int(row["TeamID"])
        w = weights[sys_name]
        team_scores[team] = team_scores.get(team, 0) + w * row["OrdinalRank"]
        team_weight_sums[team] = team_weight_sums.get(team, 0) + w

    result = []
    for team in team_scores:
        if team_weight_sums[team] > 0:
            result.append(
                {
                    "TeamID": team,
                    "massey_rank": team_scores[team] / team_weight_sums[team],
                }
            )
    return pd.DataFrame(result).set_index("TeamID")


# ── GLM Team Quality ──────────────────────────────────────────────


def _compute_glm_quality(season: int) -> pd.DataFrame:
    reg = _m_reg[_m_reg["Season"] == season]
    if len(reg) == 0:
        return pd.DataFrame(columns=["TeamID", "glm_quality"]).set_index("TeamID")

    all_teams = sorted(set(reg["WTeamID"].unique()) | set(reg["LTeamID"].unique()))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    n_games = len(reg)
    X = np.zeros((2 * n_games, n_teams), dtype=np.float32)
    y = np.zeros(2 * n_games, dtype=np.float32)

    for i, (_, g) in enumerate(reg.iterrows()):
        w_idx = team_to_idx[int(g["WTeamID"])]
        l_idx = team_to_idx[int(g["LTeamID"])]
        X[2 * i, w_idx] = 1
        X[2 * i, l_idx] = -1
        y[2 * i] = 1
        X[2 * i + 1, l_idx] = 1
        X[2 * i + 1, w_idx] = -1
        y[2 * i + 1] = 0

    lr = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", fit_intercept=False)
    lr.fit(X, y)

    quality = lr.coef_[0]
    return pd.DataFrame({"TeamID": all_teams, "glm_quality": quality}).set_index(
        "TeamID"
    )


# ── Elo Slope (seasonal momentum) ─────────────────────────────────


def _compute_elo_slope(season: int, gender: str) -> pd.DataFrame:
    """Elo trajectory: 2nd half avg minus 1st half avg (normalized)."""
    reg = _m_reg if gender == "M" else _w_reg
    reg = reg[reg["Season"] == season].sort_values("DayNum")
    if len(reg) == 0:
        return pd.DataFrame(columns=["TeamID", "elo_slope"]).set_index("TeamID")

    K, HOME_ADV_VAL = 32, 100
    elos: dict[int, float] = {}
    elo_history: dict[int, list[float]] = {}

    for _, g in reg.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        elo_w, elo_l = elos.get(w, 1500.0), elos.get(l, 1500.0)
        if g["WLoc"] == "H":
            adj_w, adj_l = elo_w + HOME_ADV_VAL, elo_l
        elif g["WLoc"] == "A":
            adj_w, adj_l = elo_w, elo_l + HOME_ADV_VAL
        else:
            adj_w, adj_l = elo_w, elo_l
        exp_w = expected_score(adj_w, adj_l)
        mov_mult = mov_multiplier(float(g["WScore"] - g["LScore"]))
        elos[w] = elo_w + K * mov_mult * (1 - exp_w)
        elos[l] = elo_l + K * mov_mult * (0 - (1 - exp_w))
        for team, elo in [(w, elos[w]), (l, elos[l])]:
            if team not in elo_history:
                elo_history[team] = []
            elo_history[team].append(elo)

    rows = []
    for team, history in elo_history.items():
        n = len(history)
        if n < 10:
            rows.append({"TeamID": team, "elo_slope": np.nan})
            continue
        mid = n // 2
        first_half = np.mean(history[:mid])
        second_half = np.mean(history[mid:])
        rows.append({"TeamID": team, "elo_slope": (second_half - first_half) / 100})
    return pd.DataFrame(rows).set_index("TeamID")


# ── Colley Matrix ──────────────────────────────────────────────────


def _compute_colley(season: int, gender: str) -> pd.DataFrame:
    """Colley Matrix method — purely win/loss based linear algebra ranking."""
    reg = _m_reg if gender == "M" else _w_reg
    reg = reg[reg["Season"] == season]
    if len(reg) == 0:
        return pd.DataFrame(columns=["TeamID", "colley_rank"]).set_index("TeamID")

    all_teams = sorted(set(reg["WTeamID"].unique()) | set(reg["LTeamID"].unique()))
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n = len(all_teams)

    # Colley matrix: C_ii = 2 + n_games_i, C_ij = -n_games_between_ij
    C = np.zeros((n, n), dtype=np.float64)
    b = np.ones(n, dtype=np.float64)

    wins = np.zeros(n, dtype=np.float64)
    losses = np.zeros(n, dtype=np.float64)
    n_games = np.zeros(n, dtype=np.float64)

    for _, g in reg.iterrows():
        w_idx = team_to_idx[int(g["WTeamID"])]
        l_idx = team_to_idx[int(g["LTeamID"])]
        C[w_idx, l_idx] -= 1
        C[l_idx, w_idx] -= 1
        n_games[w_idx] += 1
        n_games[l_idx] += 1
        wins[w_idx] += 1
        losses[l_idx] += 1

    for i in range(n):
        C[i, i] = 2 + n_games[i]
        b[i] = 1 + (wins[i] - losses[i]) / 2

    try:
        r = np.linalg.solve(C, b)
    except np.linalg.LinAlgError:
        return pd.DataFrame(columns=["TeamID", "colley_rank"]).set_index("TeamID")

    rows = [{"TeamID": all_teams[i], "colley_rank": r[i]} for i in range(n)]
    return pd.DataFrame(rows).set_index("TeamID")


# ── Non-conference SOS from Barttorvik ────────────────────────────


def _get_ncsos(season: int) -> pd.DataFrame:
    """Get ncsos (non-conference SOS) from barttorvik data."""
    from round27_pruned import read_barttorvik

    filepath = EXT / (
        "barttorvik_2026_live.csv" if season == 2026 else f"barttorvik_{season}.csv"
    )
    if not filepath.exists():
        return pd.DataFrame(columns=["TeamID", "ncsos"]).set_index("TeamID")
    bt = read_barttorvik(filepath)
    bt = bt.merge(_bt_map, on="team", how="inner")
    if "ncsos" not in bt.columns:
        return pd.DataFrame(columns=["TeamID", "ncsos"]).set_index("TeamID")
    bt = bt[["TeamID", "ncsos"]].copy().drop_duplicates(subset=["TeamID"], keep="first")
    bt["ncsos"] = pd.to_numeric(bt["ncsos"], errors="coerce")
    return bt.set_index("TeamID")


# ── Build M features ──────────────────────────────────────────────


def build_m_features(season: int) -> pd.DataFrame:
    features = build_team_features(season, "M")

    # Replace massey_rank with accuracy-weighted version
    if "massey_rank" in features.columns:
        features = features.drop(columns=["massey_rank"])
    features = features.join(_get_massey_weighted(season), how="left")

    # Add close game record
    close = _compute_close_game_record(season, margin=7)
    features = features.join(close, how="left")

    # Add GLM quality
    glm = _compute_glm_quality(season)
    features = features.join(glm, how="left")

    # Add Colley Matrix ranking
    colley = _compute_colley(season, "M")
    features = features.join(colley, how="left")

    # Add ncsos from barttorvik
    ncsos = _get_ncsos(season)
    features = features.join(ncsos, how="left")

    # Add interaction features
    if "SeedNum" in features.columns and "pt_diff" in features.columns:
        features["SeedNum_x_pt_diff"] = features["SeedNum"] * features["pt_diff"]
    if "massey_rank" in features.columns and "barthag" in features.columns:
        features["massey_rank_x_barthag"] = (
            features["massey_rank"] * features["barthag"]
        )
    if "close_win_pct" in features.columns and "pt_diff" in features.columns:
        features["close_win_pct_x_pt_diff"] = (
            features["close_win_pct"] * features["pt_diff"]
        )
    if "SeedNum" in features.columns and "massey_rank" in features.columns:
        features["SeedNum_x_massey"] = features["SeedNum"] * features["massey_rank"]
    if "ncsos" in features.columns and "massey_rank" in features.columns:
        features["ncsos_x_massey"] = features["ncsos"] * features["massey_rank"]

    return features


# ── Build W features ──────────────────────────────────────────────


def build_w_features(season: int) -> pd.DataFrame:
    features = build_team_features(season, "W")
    features = features.join(_compute_elo_slope(season, "W"), how="left")
    features = features.join(_compute_colley(season, "W"), how="left")
    # R44: Add net_rank × elo interaction
    if "net_rank" in features.columns and "elo" in features.columns:
        features["net_x_elo"] = features["net_rank"] * features["elo"]
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
            if gender == "M":
                features = build_m_features(season)
            else:
                features = build_w_features(season)
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
    print("  (R44 was: M=0.11953, W=0.13680, Combined=0.12817)")


# ── Generate submission ────────────────────────────────────────────


def generate_submission(tag: str = "r38") -> Path:
    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds: dict[str, float] = {}

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
            if gender == "M":
                features = build_m_features(season)
            else:
                features = build_w_features(season)
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
        if gender == "M":
            features_2026 = build_m_features(2026)
        else:
            features_2026 = build_w_features(2026)

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
            pred = pipe.predict_proba(x_arr)[:, 1][0]
            preds[game_id] = float(np.clip(pred, clip_low, clip_high))

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
    print("March Mania 2026 — Round 45: Drop massey_rank (M)")
    print("  M: 23 features (R44 - massey_rank), C=300")
    print("  W: 11 features (unchanged), C=0.15")
    print("=" * 70)

    evaluate_cv()
    generate_submission(tag="r45")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
