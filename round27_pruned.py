"""
March Machine Learning Mania 2026 — Round 27: Feature Pruning + Carry-over Elo

Based on R26 ablation findings:
- Carry-over Elo HELPED (Δ=-0.00043, mainly W)
- Feature selection revealed 9 M and 7 W features actively hurting LR
- Feature pruning is the highest-impact lever available

Strategy:
1. Start with carry-over Elo (confirmed winner from R26)
2. Remove features identified as hurting in R26 drop-one analysis
3. Re-run drop-one to confirm after first pruning (importance shifts)
4. Iterative backward elimination until no feature hurts
5. Re-tune C values for pruned model
6. Generate submission CSV

R25 baseline: 0.14188 | R26 combined best: 0.14141
"""

from pathlib import Path

import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXT = HERE / "external"
SUBMISSIONS = HERE / "submissions"

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]
CLIP_LOW, CLIP_HIGH = 0.01, 0.99
K_ELO = 32
HOME_ADV = 100
LAST_N = 10
N_SEEDS = 5
BASE_SEASON = 2015
ELO_CARRYOVER = 0.75

# R25 FULL feature sets (before pruning)
FULL_FEATURES_M = [
    "SeedNum",
    "win_pct",
    "pt_diff",
    "elo",
    "adjoe",
    "adjde",
    "barthag",
    "WAB",
    "sos",
    "adjt",
    "massey_rank",
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "ft_rate",
    "efg_pct_d",
    "tov_pct_d",
    "oreb_pct_d",
    "ft_rate_d",
    "ast_rate",
    "blk_pct",
    "stl_pct",
    "fg3_pct",
    "last_n_win_pct",
    "last_n_pt_diff",
    "ap_rank",
    "evanmiya_rating",
    "em_change",
    "talent",
    "exp",
    "avg_hgt",
    "ft_pct",
    "coach_tourney_exp",
]
FULL_FEATURES_W = [
    "SeedNum",
    "win_pct",
    "pt_diff",
    "elo",
    "efg_pct",
    "tov_pct",
    "oreb_pct",
    "ft_rate",
    "efg_pct_d",
    "tov_pct_d",
    "oreb_pct_d",
    "ft_rate_d",
    "ast_rate",
    "blk_pct",
    "stl_pct",
    "fg3_pct",
    "last_n_win_pct",
    "last_n_pt_diff",
    "net_rank",
]

# R26 drop-one: features that HURT (importance < -0.0003)
# These will be removed in the first pass, then we re-evaluate
M_HURT_R26 = [
    "coach_tourney_exp",  # -0.00084
    "ast_rate",  # -0.00081
    "ft_pct",  # -0.00069
    "elo",  # -0.00067 (single-season; carry-over replaces)
    "blk_pct",  # -0.00058
    "last_n_win_pct",  # -0.00058
    "ft_rate",  # -0.00042
    "talent",  # -0.00040
    "stl_pct",  # -0.00032
    "efg_pct_d",  # -0.00030
]
W_HURT_R26 = [
    "ft_rate",  # -0.00076
    "fg3_pct",  # -0.00064
    "oreb_pct",  # -0.00058
    "ast_rate",  # -0.00050
    "ft_rate_d",  # -0.00046
    "stl_pct",  # -0.00040
    "tov_pct_d",  # -0.00031
]

# ── Helpers ────────────────────────────────────────────────────────


def read_barttorvik(filepath: Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    sample = str(df["team"].iloc[0])
    if len(sample) <= 4 and sample.isupper():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=pd.errors.ParserWarning)
            df = pd.read_csv(filepath, index_col=False)
    return df


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400))


def mov_multiplier(margin: float) -> float:
    return np.log(abs(margin) + 1) * (2.2 / (0.001 * abs(margin) + 2.2))


# ── Load static data once ────────────────────────────────────────

print("Loading static data...")
_m_reg = pd.read_csv(DATA / "MRegularSeasonCompactResults.csv")
_w_reg = pd.read_csv(DATA / "WRegularSeasonCompactResults.csv")
_m_det = pd.read_csv(DATA / "MRegularSeasonDetailedResults.csv")
_w_det = pd.read_csv(DATA / "WRegularSeasonDetailedResults.csv")
_m_seeds = pd.read_csv(DATA / "MNCAATourneySeeds.csv")
_w_seeds = pd.read_csv(DATA / "WNCAATourneySeeds.csv")
for _s in [_m_seeds, _w_seeds]:
    _s["SeedNum"] = _s["Seed"].str.extract(r"(\d+)").astype(int)
_m_tourney = pd.read_csv(DATA / "MNCAATourneyCompactResults.csv")
_w_tourney = pd.read_csv(DATA / "WNCAATourneyCompactResults.csv")
_bt_map = pd.read_csv(EXT / "barttorvik_team_id_map.csv")
print("Loading Massey Ordinals...")
_massey = pd.read_csv(DATA / "MMasseyOrdinals.csv")

_evan_raw = pd.read_csv(EXT / "EvanMiya.csv")
_preseason_raw = pd.read_csv(EXT / "KenPom Preseason.csv")
_nishaanamin_map = pd.read_csv(EXT / "team_id_map.csv")

_kb_raw = pd.read_csv(EXT / "KenPom Barttorvik.csv")
_kb_talent_exp_joined = (
    _kb_raw[["YEAR", "TEAM", "TALENT", "EXP"]]
    .rename(columns={"YEAR": "Season", "TALENT": "talent", "EXP": "exp"})
    .merge(_nishaanamin_map, on="TEAM", how="inner")
    .drop(columns=["TEAM"])
)
_kb_hgt_joined = (
    _kb_raw[["YEAR", "TEAM", "AVG HGT"]]
    .rename(columns={"YEAR": "Season", "AVG HGT": "avg_hgt"})
    .merge(_nishaanamin_map, on="TEAM", how="inner")
    .drop(columns=["TEAM"])
)
_kb_ft_joined = (
    _kb_raw[["YEAR", "TEAM", "FT%"]]
    .rename(columns={"YEAR": "Season", "FT%": "ft_pct"})
    .merge(_nishaanamin_map, on="TEAM", how="inner")
    .drop(columns=["TEAM"])
)
_evan_joined = (
    _evan_raw[["YEAR", "TEAM", "RELATIVE RATING"]]
    .rename(columns={"YEAR": "Season", "RELATIVE RATING": "evanmiya_rating"})
    .merge(_nishaanamin_map, on="TEAM", how="inner")
    .drop(columns=["TEAM"])
)
_preseason_joined = (
    _preseason_raw[["YEAR", "TEAM", "KADJ EM CHANGE"]]
    .rename(columns={"YEAR": "Season", "KADJ EM CHANGE": "em_change"})
    .merge(_nishaanamin_map, on="TEAM", how="inner")
    .drop(columns=["TEAM"])
)

_m_coaches = pd.read_csv(DATA / "MTeamCoaches.csv")
_ap_poll_raw = pd.read_csv(EXT / "AP Poll Data.csv")
_ap_team_map = pd.read_csv(EXT / "team_id_map.csv")
_ap_w1 = (
    _ap_poll_raw[_ap_poll_raw["WEEK"] == 1][["YEAR", "TEAM", "AP RANK"]]
    .rename(columns={"YEAR": "Season", "AP RANK": "ap_rank"})
    .merge(_ap_team_map, on="TEAM", how="inner")
    .drop(columns=["TEAM"])
)
_wnet = pd.read_csv(EXT / "wncaa_net" / "WNET_2026.csv")


# ── Carry-over Elo ────────────────────────────────────────────────


def _precompute_all_elos(gender: str) -> dict[int, dict[int, float]]:
    """Compute Elo for every team in every season, carrying over across years."""
    reg = _m_reg if gender == "M" else _w_reg
    all_seasons = sorted(reg["Season"].unique())
    prev_elos: dict[int, float] = {}
    season_elos: dict[int, dict[int, float]] = {}

    for season in all_seasons:
        elos: dict[int, float] = {}
        for team_id, elo in prev_elos.items():
            elos[team_id] = ELO_CARRYOVER * elo + (1 - ELO_CARRYOVER) * 1500.0

        season_games = reg[reg["Season"] == season].sort_values("DayNum")
        for _, g in season_games.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            elo_w, elo_l = elos.get(w, 1500.0), elos.get(l, 1500.0)
            if g["WLoc"] == "H":
                adj_w, adj_l = elo_w + HOME_ADV, elo_l
            elif g["WLoc"] == "A":
                adj_w, adj_l = elo_w, elo_l + HOME_ADV
            else:
                adj_w, adj_l = elo_w, elo_l
            exp_w = expected_score(adj_w, adj_l)
            mov_mult = mov_multiplier(float(g["WScore"] - g["LScore"]))
            elos[w] = elo_w + K_ELO * mov_mult * (1 - exp_w)
            elos[l] = elo_l + K_ELO * mov_mult * (0 - (1 - exp_w))

        season_elos[season] = dict(elos)
        prev_elos = dict(elos)

    return season_elos


print("Pre-computing carry-over Elos...")
_carryover_elos_m = _precompute_all_elos("M")
_carryover_elos_w = _precompute_all_elos("W")


def _compute_carryover_elo(season: int, gender: str) -> pd.DataFrame:
    elos = _carryover_elos_m if gender == "M" else _carryover_elos_w
    if season not in elos:
        return pd.DataFrame(columns=["TeamID", "elo"]).set_index("TeamID")
    d = elos[season]
    return pd.DataFrame({"TeamID": list(d.keys()), "elo": list(d.values())}).set_index(
        "TeamID"
    )


# ── Feature Builders ─────────────────────────────────────────────


def _compute_reg_season_stats(season: int, gender: str) -> pd.DataFrame:
    reg = _m_reg if gender == "M" else _w_reg
    reg = reg[reg["Season"] == season]
    w_rows = reg[["WTeamID", "WScore", "LScore"]].copy()
    w_rows.columns = ["TeamID", "PtsFor", "PtsAgainst"]
    w_rows["Win"] = 1
    l_rows = reg[["LTeamID", "LScore", "WScore"]].copy()
    l_rows.columns = ["TeamID", "PtsFor", "PtsAgainst"]
    l_rows["Win"] = 0
    all_games = pd.concat([w_rows, l_rows], ignore_index=True)
    stats = all_games.groupby("TeamID").agg(
        num_games=("Win", "count"),
        num_wins=("Win", "sum"),
        pts_for=("PtsFor", "sum"),
        pts_against=("PtsAgainst", "sum"),
    )
    stats["win_pct"] = stats["num_wins"] / stats["num_games"]
    stats["pt_diff"] = (stats["pts_for"] - stats["pts_against"]) / stats["num_games"]
    return stats[["win_pct", "pt_diff"]]


def _compute_last_n_games(season: int, gender: str) -> pd.DataFrame:
    reg = _m_reg if gender == "M" else _w_reg
    reg = reg[reg["Season"] == season].sort_values("DayNum")
    w_rows = pd.DataFrame(
        {
            "DayNum": reg["DayNum"],
            "TeamID": reg["WTeamID"],
            "Win": 1,
            "PtDiff": reg["WScore"] - reg["LScore"],
        }
    )
    l_rows = pd.DataFrame(
        {
            "DayNum": reg["DayNum"],
            "TeamID": reg["LTeamID"],
            "Win": 0,
            "PtDiff": reg["LScore"] - reg["WScore"],
        }
    )
    all_games = pd.concat([w_rows, l_rows], ignore_index=True).sort_values("DayNum")
    results = {}
    for team_id, team_games in all_games.groupby("TeamID"):
        last_n = team_games.tail(LAST_N)
        results[team_id] = {
            "last_n_win_pct": last_n["Win"].mean(),
            "last_n_pt_diff": last_n["PtDiff"].mean(),
        }
    return pd.DataFrame(results).T.rename_axis("TeamID")


def _compute_four_factors(season: int, gender: str) -> pd.DataFrame:
    det = _m_det if gender == "M" else _w_det
    det = det[det["Season"] == season]
    cols = [
        "efg_pct",
        "tov_pct",
        "oreb_pct",
        "ft_rate",
        "efg_pct_d",
        "tov_pct_d",
        "oreb_pct_d",
        "ft_rate_d",
        "ast_rate",
        "blk_pct",
        "stl_pct",
        "fg3_pct",
    ]
    if len(det) == 0:
        return pd.DataFrame(columns=["TeamID"] + cols).set_index("TeamID")
    w_off = pd.DataFrame(
        {
            "TeamID": det["WTeamID"],
            "FGM": det["WFGM"],
            "FGA": det["WFGA"],
            "FGM3": det["WFGM3"],
            "FGA3": det["WFGA3"],
            "FTM": det["WFTM"],
            "FTA": det["WFTA"],
            "OR": det["WOR"],
            "DR": det["WDR"],
            "Ast": det["WAst"],
            "TO": det["WTO"],
            "Stl": det["WStl"],
            "Blk": det["WBlk"],
            "OppDR": det["LDR"],
            "OppFGM": det["LFGM"],
            "OppFGA": det["LFGA"],
            "OppFGM3": det["LFGM3"],
            "OppFGA3": det["LFGA3"],
            "OppFTA": det["LFTA"],
            "OppOR": det["LOR"],
            "OppTO": det["LTO"],
        }
    )
    l_off = pd.DataFrame(
        {
            "TeamID": det["LTeamID"],
            "FGM": det["LFGM"],
            "FGA": det["LFGA"],
            "FGM3": det["LFGM3"],
            "FGA3": det["LFGA3"],
            "FTM": det["LFTM"],
            "FTA": det["LFTA"],
            "OR": det["LOR"],
            "DR": det["LDR"],
            "Ast": det["LAst"],
            "TO": det["LTO"],
            "Stl": det["LStl"],
            "Blk": det["LBlk"],
            "OppDR": det["WDR"],
            "OppFGM": det["WFGM"],
            "OppFGA": det["WFGA"],
            "OppFGM3": det["WFGM3"],
            "OppFGA3": det["WFGA3"],
            "OppFTA": det["WFTA"],
            "OppOR": det["WOR"],
            "OppTO": det["WTO"],
        }
    )
    all_games = pd.concat([w_off, l_off], ignore_index=True)
    agg = all_games.groupby("TeamID").sum()
    agg["efg_pct"] = (agg["FGM"] + 0.5 * agg["FGM3"]) / agg["FGA"]
    agg["tov_pct"] = agg["TO"] / (agg["FGA"] + 0.44 * agg["FTA"] + agg["TO"])
    agg["oreb_pct"] = agg["OR"] / (agg["OR"] + agg["OppDR"])
    agg["ft_rate"] = agg["FTA"] / agg["FGA"]
    agg["efg_pct_d"] = (agg["OppFGM"] + 0.5 * agg["OppFGM3"]) / agg["OppFGA"]
    agg["tov_pct_d"] = agg["OppTO"] / (
        agg["OppFGA"] + 0.44 * agg["OppFTA"] + agg["OppTO"]
    )
    agg["oreb_pct_d"] = agg["OppOR"] / (agg["OppOR"] + agg["DR"])
    agg["ft_rate_d"] = agg["OppFTA"] / agg["OppFGA"]
    n_games = all_games.groupby("TeamID")["FGA"].count()
    agg["ast_rate"] = agg["Ast"] / n_games
    agg["blk_pct"] = agg["Blk"] / agg["OppFGA"]
    agg["stl_pct"] = agg["Stl"] / n_games
    agg["fg3_pct"] = agg["FGM3"] / agg["FGA3"].replace(0, np.nan)
    return agg[cols]


def _get_seeds(season: int, gender: str) -> pd.DataFrame:
    seeds = _m_seeds if gender == "M" else _w_seeds
    return seeds[seeds["Season"] == season][["TeamID", "SeedNum"]].set_index("TeamID")


def _get_barttorvik(season: int) -> pd.DataFrame:
    filepath = EXT / (
        "barttorvik_2026_live.csv" if season == 2026 else f"barttorvik_{season}.csv"
    )
    bt_cols = ["adjoe", "adjde", "barthag", "WAB", "sos", "adjt", "Qual D", "Qual O"]
    if not filepath.exists():
        return pd.DataFrame(columns=["TeamID"] + bt_cols).set_index("TeamID")
    bt = read_barttorvik(filepath)
    bt = bt.merge(_bt_map, on="team", how="inner")
    available = ["TeamID"] + [c for c in bt_cols if c in bt.columns]
    bt = bt[available].copy().drop_duplicates(subset=["TeamID"], keep="first")
    for col in bt_cols:
        bt[col] = pd.to_numeric(bt.get(col, np.nan), errors="coerce")
    return bt.set_index("TeamID")


def _get_massey_composite(season: int) -> pd.DataFrame:
    systems = [
        "POM",
        "MOR",
        "KPK",
        "NET",
        "WAB",
        "MAS",
        "EBP",
        "LOG",
        "TRP",
        "DOK",
        "DCI",
        "TRK",
        "DOL",
        "WLK",
        "BWE",
    ]
    m = _massey[(_massey["Season"] == season) & (_massey["SystemName"].isin(systems))]
    if len(m) == 0:
        return pd.DataFrame(columns=["TeamID", "massey_rank"]).set_index("TeamID")
    latest = m.groupby(["SystemName", "TeamID"])["RankingDayNum"].max().reset_index()
    latest = latest.merge(m, on=["SystemName", "TeamID", "RankingDayNum"])
    avg_rank = latest.groupby("TeamID")["OrdinalRank"].mean().reset_index()
    avg_rank.columns = ["TeamID", "massey_rank"]
    return avg_rank.set_index("TeamID")


def _get_evanmiya(season: int) -> pd.DataFrame:
    rows = _evan_joined[_evan_joined["Season"] == season][
        ["TeamID", "evanmiya_rating"]
    ].copy()
    rows["evanmiya_rating"] = pd.to_numeric(rows["evanmiya_rating"], errors="coerce")
    return rows.drop_duplicates("TeamID").set_index("TeamID")


def _get_em_change(season: int) -> pd.DataFrame:
    rows = _preseason_joined[_preseason_joined["Season"] == season][
        ["TeamID", "em_change"]
    ].copy()
    rows["em_change"] = pd.to_numeric(rows["em_change"], errors="coerce")
    return rows.drop_duplicates("TeamID").set_index("TeamID")


def _get_talent_exp(season: int) -> pd.DataFrame:
    rows = _kb_talent_exp_joined[_kb_talent_exp_joined["Season"] == season][
        ["TeamID", "talent", "exp"]
    ].copy()
    rows["talent"] = pd.to_numeric(rows["talent"], errors="coerce")
    rows["exp"] = pd.to_numeric(rows["exp"], errors="coerce")
    return rows.drop_duplicates("TeamID").set_index("TeamID")


def _get_avg_hgt(season: int) -> pd.DataFrame:
    rows = _kb_hgt_joined[_kb_hgt_joined["Season"] == season][
        ["TeamID", "avg_hgt"]
    ].copy()
    rows["avg_hgt"] = pd.to_numeric(rows["avg_hgt"], errors="coerce")
    return rows.drop_duplicates("TeamID").set_index("TeamID")


def _get_ft_pct(season: int) -> pd.DataFrame:
    rows = _kb_ft_joined[_kb_ft_joined["Season"] == season][["TeamID", "ft_pct"]].copy()
    rows["ft_pct"] = pd.to_numeric(rows["ft_pct"], errors="coerce")
    return rows.drop_duplicates("TeamID").set_index("TeamID")


def _get_coach_tourney_exp(season: int) -> pd.DataFrame:
    tourney_team_seasons: set[tuple[int, int]] = set()
    for _, g in _m_tourney.iterrows():
        s = int(g["Season"])
        tourney_team_seasons.add((s, int(g["WTeamID"])))
        tourney_team_seasons.add((s, int(g["LTeamID"])))
    tourney_coaches = _m_coaches[_m_coaches["LastDayNum"] >= 134].copy()
    coach_exp: dict[str, int] = {}
    result_rows = []
    for s in sorted(tourney_coaches["Season"].unique()):
        season_coaches = tourney_coaches[tourney_coaches["Season"] == s]
        for _, row in season_coaches.iterrows():
            coach = row["CoachName"]
            team_id = int(row["TeamID"])
            if s == season:
                result_rows.append(
                    {"TeamID": team_id, "coach_tourney_exp": coach_exp.get(coach, 0)}
                )
            if (int(s), team_id) in tourney_team_seasons:
                coach_exp[coach] = coach_exp.get(coach, 0) + 1
    if not result_rows:
        return pd.DataFrame(columns=["TeamID", "coach_tourney_exp"]).set_index("TeamID")
    return pd.DataFrame(result_rows).drop_duplicates("TeamID").set_index("TeamID")


def _get_ap_rank(season: int) -> pd.DataFrame:
    season_ap = _ap_w1[_ap_w1["Season"] == season][["TeamID", "ap_rank"]].copy()
    season_ap["ap_rank"] = pd.to_numeric(season_ap["ap_rank"], errors="coerce")
    return season_ap.drop_duplicates("TeamID").set_index("TeamID")


def _get_wnet(season: int) -> pd.DataFrame:
    w = _wnet[_wnet["Season"] == season][["TeamID", "Rank"]].copy()
    w = w.rename(columns={"Rank": "net_rank"})
    return w.drop_duplicates("TeamID").set_index("TeamID")


# ── Team features builder ────────────────────────────────────────


def build_team_features(season: int, gender: str) -> pd.DataFrame:
    """Always uses carry-over Elo (R26 confirmed winner)."""
    features = (
        _compute_reg_season_stats(season, gender)
        .join(_compute_carryover_elo(season, gender), how="outer")
        .join(_get_seeds(season, gender), how="outer")
        .join(_compute_four_factors(season, gender), how="left")
        .join(_compute_last_n_games(season, gender), how="left")
    )
    if gender == "M":
        features = features.join(_get_barttorvik(season), how="left")
        features = features.join(_get_massey_composite(season), how="left")
        features = features.join(_get_ap_rank(season), how="left")
        features = features.join(_get_evanmiya(season), how="left")
        features = features.join(_get_em_change(season), how="left")
        features = features.join(_get_talent_exp(season), how="left")
        features = features.join(_get_avg_hgt(season), how="left")
        features = features.join(_get_ft_pct(season), how="left")
        features = features.join(_get_coach_tourney_exp(season), how="left")
    else:
        features = features.join(_get_wnet(season), how="left")
    features["SeedNum"] = features["SeedNum"].fillna(17)
    features["elo"] = features["elo"].fillna(1500.0)
    return features


# ── Training Set Builder ─────────────────────────────────────────


def build_training_set(
    diff_feats_m: list[str],
    diff_feats_w: list[str],
) -> dict[str, tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, list[str]]]:
    result = {}
    for gender, tourney, diff_feats in [
        ("M", _m_tourney, diff_feats_m),
        ("W", _w_tourney, diff_feats_w),
    ]:
        rows = []
        for season in TRAIN_SEASONS:
            games = tourney[tourney["Season"] == season]
            if len(games) == 0:
                continue
            features = build_team_features(season, gender)
            for _, g in games.iterrows():
                t1 = min(int(g["WTeamID"]), int(g["LTeamID"]))
                t2 = max(int(g["WTeamID"]), int(g["LTeamID"]))
                row: dict[str, float] = {
                    "Season": season,
                    "Team1ID": t1,
                    "Team2ID": t2,
                    "Team1Win": 1 if int(g["WTeamID"]) == t1 else 0,
                }
                for feat in diff_feats:
                    v1 = features.loc[t1, feat] if t1 in features.index else np.nan
                    v2 = features.loc[t2, feat] if t2 in features.index else np.nan
                    row[f"d_{feat}"] = v1 - v2
                rows.append(row)
        df = pd.DataFrame(rows)
        feat_cols = [f"d_{f}" for f in diff_feats]
        X = df[feat_cols].values.astype(np.float32)
        y = df["Team1Win"].values.astype(np.float32)
        groups = df["Season"].values.astype(int)
        result[gender] = (df, X, y, groups, feat_cols)
    return result


# ── LR LOSO CV ───────────────────────────────────────────────────


def cv_lr(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    C: float = 1.0,
    clip_low: float = CLIP_LOW,
    clip_high: float = CLIP_HIGH,
) -> tuple[float, np.ndarray]:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=len(set(groups)))
    oof = np.zeros(len(y))
    for train_idx, val_idx in gkf.split(X, y, groups):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        oof[val_idx] = np.clip(
            pipe.predict_proba(X[val_idx])[:, 1], clip_low, clip_high
        )
    return brier_score_loss(y, oof), oof


# ── Drop-one feature importance ──────────────────────────────────


def drop_one_importance(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_cols: list[str],
    C: float,
) -> list[tuple[str, float]]:
    """Drop each feature, measure CV change. Positive = feature helps."""
    base_brier, _ = cv_lr(X, y, groups, C=C)
    importances = []
    for i, col in enumerate(feat_cols):
        X_dropped = np.delete(X, i, axis=1)
        dropped_brier, _ = cv_lr(X_dropped, y, groups, C=C)
        importances.append((col, dropped_brier - base_brier))
    return sorted(importances, key=lambda x: x[1], reverse=True)


# ── Iterative backward elimination ───────────────────────────────


def iterative_backward_elimination(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_cols: list[str],
    C: float,
    threshold: float = 0.0,
    verbose: bool = True,
) -> tuple[list[str], float]:
    """Iteratively remove the most harmful feature until none hurt."""
    current_feats = list(feat_cols)
    current_X = X.copy()
    current_brier, _ = cv_lr(current_X, y, groups, C=C)

    if verbose:
        print(f"    Start: {len(current_feats)} features, Brier={current_brier:.5f}")

    iteration = 0
    while len(current_feats) > 1:
        iteration += 1
        imps = drop_one_importance(current_X, y, groups, current_feats, C)
        worst_feat, worst_imp = imps[-1]

        if worst_imp >= threshold:
            if verbose:
                print(
                    f"    Iter {iteration}: No feature hurts (worst: {worst_feat} = {worst_imp:+.5f}). Done."
                )
            break

        # Remove worst feature
        idx = current_feats.index(worst_feat)
        current_feats.pop(idx)
        current_X = np.delete(current_X, idx, axis=1)
        new_brier, _ = cv_lr(current_X, y, groups, C=C)

        if verbose:
            print(
                f"    Iter {iteration}: Drop {worst_feat} ({worst_imp:+.5f}): {current_brier:.5f} → {new_brier:.5f} (Δ={new_brier - current_brier:+.5f})"
            )

        current_brier = new_brier

    return current_feats, current_brier


# ── Submission generation ────────────────────────────────────────


def generate_submission(
    feats_m: list[str],
    feats_w: list[str],
    C_m: float,
    C_w: float,
    tag: str = "r27",
) -> Path:
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds = {}

    for gender, diff_feats, C_val in [("M", feats_m, C_m), ("W", feats_w, C_w)]:
        tourney = _m_tourney if gender == "M" else _w_tourney
        # Build training set from all seasons
        rows = []
        for season in TRAIN_SEASONS:
            games = tourney[tourney["Season"] == season]
            if len(games) == 0:
                continue
            features = build_team_features(season, gender)
            for _, g in games.iterrows():
                t1 = min(int(g["WTeamID"]), int(g["LTeamID"]))
                t2 = max(int(g["WTeamID"]), int(g["LTeamID"]))
                row: dict[str, float] = {
                    "Team1Win": 1 if int(g["WTeamID"]) == t1 else 0,
                }
                for feat in diff_feats:
                    v1 = features.loc[t1, feat] if t1 in features.index else np.nan
                    v2 = features.loc[t2, feat] if t2 in features.index else np.nan
                    row[f"d_{feat}"] = v1 - v2
                rows.append(row)

        df = pd.DataFrame(rows)
        feat_cols = [f"d_{f}" for f in diff_feats]
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

        # Predict 2026 matchups
        features_2026 = build_team_features(2026, gender)
        prefix = "2026_" if gender == "M" else "2026_"
        for _, row in sub.iterrows():
            game_id = row["ID"]
            parts = game_id.split("_")
            if len(parts) != 3:
                continue
            season_str, t1_str, t2_str = parts
            if int(season_str) != 2026:
                continue
            t1, t2 = int(t1_str), int(t2_str)
            # Determine gender by team ID range
            is_mens = t1 < 3000
            if (gender == "M" and not is_mens) or (gender == "W" and is_mens):
                continue

            x_row = []
            for feat in diff_feats:
                v1 = (
                    features_2026.loc[t1, feat] if t1 in features_2026.index else np.nan
                )
                v2 = (
                    features_2026.loc[t2, feat] if t2 in features_2026.index else np.nan
                )
                x_row.append(v1 - v2 if not (np.isnan(v1) or np.isnan(v2)) else np.nan)

            x_arr = np.array([x_row], dtype=np.float32)
            pred = pipe.predict_proba(x_arr)[:, 1][0]
            pred = np.clip(pred, CLIP_LOW, CLIP_HIGH)
            preds[game_id] = pred

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
    print("March Mania 2026 — Round 27: Feature Pruning + Carry-over Elo")
    print("=" * 70)

    R25_CV = 0.14188
    R26_COMBINED = 0.14141

    # ── Step 1: Baseline with carry-over Elo (R26 winner) ──
    print("\n── Step 1: Carry-over Elo baseline (all features) ──")
    data = build_training_set(FULL_FEATURES_M, FULL_FEATURES_W)
    n_m, n_w = len(data["M"][2]), len(data["W"][2])

    BEST_C = {"M": 10.0, "W": 0.3}
    b_m, _ = cv_lr(data["M"][1], data["M"][2], data["M"][3], C=BEST_C["M"])
    b_w, _ = cv_lr(data["W"][1], data["W"][2], data["W"][3], C=BEST_C["W"])
    baseline = (b_m * n_m + b_w * n_w) / (n_m + n_w)
    print(
        f"  M: {b_m:.5f} ({len(data['M'][4])} feats)  W: {b_w:.5f} ({len(data['W'][4])} feats)"
    )
    print(f"  Combined: {baseline:.5f}")

    # ── Step 2: Quick prune — remove R26-identified hurting features ──
    print("\n── Step 2: Quick prune (remove R26 hurting features) ──")
    pruned_m = [f for f in FULL_FEATURES_M if f not in M_HURT_R26]
    pruned_w = [f for f in FULL_FEATURES_W if f not in W_HURT_R26]
    print(
        f"  M: {len(FULL_FEATURES_M)} → {len(pruned_m)} features (dropped {len(M_HURT_R26)})"
    )
    print(
        f"  W: {len(FULL_FEATURES_W)} → {len(pruned_w)} features (dropped {len(W_HURT_R26)})"
    )

    data_p = build_training_set(pruned_m, pruned_w)
    bp_m, _ = cv_lr(data_p["M"][1], data_p["M"][2], data_p["M"][3], C=BEST_C["M"])
    bp_w, _ = cv_lr(data_p["W"][1], data_p["W"][2], data_p["W"][3], C=BEST_C["W"])
    pruned_combined = (bp_m * n_m + bp_w * n_w) / (n_m + n_w)
    print(f"  M: {bp_m:.5f}  W: {bp_w:.5f}  Combined: {pruned_combined:.5f}")
    print(f"  Δ vs carryover baseline: {pruned_combined - baseline:+.5f}")
    print(f"  Δ vs R25: {pruned_combined - R25_CV:+.5f}")

    # ── Step 3: Re-run drop-one on pruned features (importance shifts) ──
    print("\n── Step 3: Drop-one analysis on pruned features ──")
    for g in ["M", "W"]:
        _, X, y, groups, fc = data_p[g]
        print(f"\n  {g} drop-one ({len(fc)} features, C={BEST_C[g]}):")
        imps = drop_one_importance(X, y, groups, fc, C=BEST_C[g])
        for name, imp in imps:
            marker = " ← HURTS" if imp < -0.0001 else ""
            print(f"    {name:>25s}: {imp:+.5f}{marker}")

    # ── Step 4: Iterative backward elimination ──
    print("\n── Step 4: Iterative backward elimination ──")
    final_feats = {}
    final_briers = {}
    for g in ["M", "W"]:
        _, X, y, groups, fc = data_p[g]
        print(f"\n  {g}:")
        best_feats, best_brier = iterative_backward_elimination(
            X, y, groups, fc, C=BEST_C[g], threshold=0.0, verbose=True
        )
        final_feats[g] = best_feats
        final_briers[g] = best_brier

    # Convert back to original feature names (strip d_ prefix)
    final_m = [f[2:] for f in final_feats["M"]]
    final_w = [f[2:] for f in final_feats["W"]]
    print(f"\n  Final M features ({len(final_m)}): {final_m}")
    print(f"  Final W features ({len(final_w)}): {final_w}")

    # ── Step 5: Re-tune C for final feature sets ──
    print("\n── Step 5: Re-tune C for final feature sets ──")
    data_final = build_training_set(final_m, final_w)
    best_C_final: dict[str, float] = {}
    for g in ["M", "W"]:
        _, X, y, groups, fc = data_final[g]
        best_c, best_b = 1.0, float("inf")
        c_range = [
            0.01,
            0.03,
            0.05,
            0.1,
            0.2,
            0.3,
            0.5,
            0.7,
            1.0,
            2.0,
            3.0,
            5.0,
            7.0,
            10.0,
            15.0,
            20.0,
            30.0,
        ]
        for c in c_range:
            brier, _ = cv_lr(X, y, groups, C=c)
            print(f"    {g} C={c:6.2f}: Brier={brier:.5f}")
            if brier < best_b:
                best_b = brier
                best_c = c
        best_C_final[g] = best_c
        print(f"  → {g} best: C={best_c}, Brier={best_b:.5f}")

    # Combined score
    bf_m, _ = cv_lr(
        data_final["M"][1], data_final["M"][2], data_final["M"][3], C=best_C_final["M"]
    )
    bf_w, _ = cv_lr(
        data_final["W"][1], data_final["W"][2], data_final["W"][3], C=best_C_final["W"]
    )
    final_combined = (bf_m * n_m + bf_w * n_w) / (n_m + n_w)

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  R25 baseline:          {R25_CV:.5f}")
    print(f"  R26 combined:          {R26_COMBINED:.5f}")
    print(
        f"  R27 quick prune:       {pruned_combined:.5f}  (Δ vs R25: {pruned_combined - R25_CV:+.5f})"
    )
    print(
        f"  R27 iterative prune:   {final_combined:.5f}  (Δ vs R25: {final_combined - R25_CV:+.5f})"
    )
    print(f"  M: Brier={bf_m:.5f}, C={best_C_final['M']}, {len(final_m)} features")
    print(f"  W: Brier={bf_w:.5f}, C={best_C_final['W']}, {len(final_w)} features")
    print(f"  M features: {final_m}")
    print(f"  W features: {final_w}")
    print("=" * 70)

    # ── Generate submission ──
    print("\n── Generating submission ──")
    generate_submission(
        final_m, final_w, best_C_final["M"], best_C_final["W"], tag="r27_pruned"
    )
