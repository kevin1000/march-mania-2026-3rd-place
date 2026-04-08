"""
March Machine Learning Mania 2026 — Round 52: Triple-Market Blend

Strategy: LR model + triple-market blend (ESPN BPI + Vegas + Kalshi)
  - R51 features: R50 + coach_pase (25 M feats, C=100)
  - Qual D/O tested but HURT — not included
  - Vegas moneylines converted to no-vig implied probabilities
  - Market consensus = average(BPI, Vegas) for R1 games
  - Market consensus = 60% Vegas + 40% BPI (Vegas reacts faster to injuries)
  - R51 CV: M=0.11199, W=0.13506, Combined=0.12352

Blend tiers:
  M R1:    20% LR + 80% market_consensus (60% Vegas + 40% BPI)
  M other: 60% LR + 25% BPI_BT + 15% Kalshi_BT
  W R1:    35% LR + 65% market_consensus (60% Vegas + 40% BPI)
  W other: 70% LR + 15% BPI_BT + 15% Kalshi_BT
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
from round50_final import (
    build_w_features,
    FEATURES_W,
    CLIP_M_LOW,
    CLIP_M_HIGH,
    CLIP_W_LOW,
    CLIP_W_HIGH,
    BPI_TEAM_MAP,
    BPI_M_CHAMP,
    BPI_R1_GAMES,
    bpi_bt_pairwise,
)
from round51_final import (
    build_m_features,  # R51: adds coach_pase
    BPI_W_TEAM_MAP,
    BPI_W_CHAMP,
    BPI_W_R1_GAMES,
    bpi_w_bt_pairwise,
)

# R51 results: +coach_pase, C=100 (was 500 in R50)
# M CV: 0.11199 (was 0.11541), Combined: 0.12352 (was 0.12524)
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
    "coach_pase",  # NEW in R51: Δ=-0.00331
]
C_M = 100.0  # R51: retuned from 500 with coach_pase
C_W = 0.15  # unchanged from R50
from round46_final import (
    load_kalshi_odds,
    kalshi_pairwise,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SUBMISSIONS = HERE / "submissions"

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]


# ── Vegas Moneyline Data (scraped March 19, 2026) ──────────────
#
# Converted from moneyline to no-vig implied probability:
#   Fav: p = |ML| / (|ML| + 100)
#   Dog: p = 100 / (ML + 100)
#   No-vig: normalize so both sum to 1.0
#
# Format: (favored_team, underdog_team, favored_no_vig_prob)

_VEGAS_M_R1_RAW: list[tuple[str, str, float]] = [
    # Thursday games
    ("Ohio State", "TCU", 0.564),  # OSU -142 / TCU +120
    ("Nebraska", "Troy", 0.872),  # NEB -1000 / Troy +650
    ("Louisville", "South Florida", 0.664),  # LOU -225 / USF +185
    ("Wisconsin", "High Point", 0.800),  # WIS -500 / HP +380
    ("Duke", "Siena", 0.973),  # Duke -20000 / Siena +3500
    ("Vanderbilt", "McNeese", 0.840),  # VANDY -700 / McNeese +500
    ("Michigan State", "North Dakota State", 0.912),  # MSU -1800 / NDSU +1000
    ("Arkansas", "Hawaii", 0.899),  # ARK -1450 / Hawaii +850
    ("North Carolina", "VCU", 0.564),  # UNC -142 / VCU +120
    ("Michigan", "Howard", 0.976),  # MICH -50000 / Howard +4000
    ("BYU", "Texas", 0.572),  # BYU -148 / Texas +124
    ("Saint Mary's", "Texas A&M", 0.593),  # SMC -162 / TAMU +136
    ("Illinois", "Pennsylvania", 0.954),  # ILL -6500 / Penn +2000
    ("Georgia", "Saint Louis", 0.572),  # UGA -148 / SLU +124
    ("Gonzaga", "Kennesaw State", 0.936),  # GONZ -3200 / KSU +1400
    ("Houston", "Idaho", 0.958),  # HOU -8000 / Idaho +2200
    # Friday games
    ("Kentucky", "Santa Clara", 0.593),  # UK -162 / SC +136
    ("Texas Tech", "Akron", 0.741),  # TTU -340 / Akron +270
    ("Arizona", "Long Island University", 0.981),  # AZ -100000 / LIU +5000
    ("Virginia", "Wright State", 0.931),  # UVA -2800 / WSU +1300
    ("Iowa State", "Tennessee State", 0.958),  # ISU -8000 / TSU +2200
    ("Alabama", "Hofstra", 0.853),  # BAMA -800 / Hofstra +550
    ("Utah State", "Villanova", 0.532),  # USU -125 / NOVA +105
    ("Iowa", "Clemson", 0.551),  # IOWA -135 / CLEM +114
    ("St. John's", "Northern Iowa", 0.822),  # SJU -600 / UNI +440
    ("UCLA", "UCF", 0.686),  # UCLA -250 / UCF +205
    ("Purdue", "Queens", 0.958),  # PUR -8000 / Queens +2200
    ("Kansas", "California Baptist", 0.887),  # KU -1200 / CBU +750
    ("UConn", "Furman", 0.946),  # UCONN -4500 / Furman +1700
    ("Miami", "Missouri", 0.543),  # MIA -130 / MIZ +110
]

# Additional team mappings needed for Vegas data
_VEGAS_EXTRA_MAP: dict[str, int] = {
    "Santa Clara": 1365,
    "TCU": 1395,
}


def _build_vegas_r1_games() -> dict[tuple[int, int], float]:
    """Convert Vegas R1 moneyline predictions to {(min_id, max_id): P(min_id wins)}."""
    # Merge BPI_TEAM_MAP with extra
    full_map = {**BPI_TEAM_MAP, **_VEGAS_EXTRA_MAP}
    games: dict[tuple[int, int], float] = {}
    for fav, dog, fav_prob in _VEGAS_M_R1_RAW:
        fav_id = full_map.get(fav)
        dog_id = full_map.get(dog)
        if fav_id is None or dog_id is None:
            print(
                f"  WARNING: unmapped Vegas team: {fav} ({fav_id}) or {dog} ({dog_id})"
            )
            continue
        t1 = min(fav_id, dog_id)
        t2 = max(fav_id, dog_id)
        if t1 == fav_id:
            games[(t1, t2)] = fav_prob
        else:
            games[(t1, t2)] = 1.0 - fav_prob
    return games


VEGAS_M_R1_GAMES = _build_vegas_r1_games()


# ── Women's Vegas Moneyline Data (ESPN, March 18-19, 2026) ─────
# No-vig implied probabilities from moneylines
# Note: NC State is Vegas favorite over Tennessee (BPI disagrees!)

_VEGAS_W_R1_RAW: list[tuple[str, str, float]] = [
    # Games with real moneylines (16 games)
    ("Oregon", "Virginia Tech", 0.604),  # -170 / +142
    ("Washington", "South Dakota State", 0.696),  # -265 / +215
    ("Maryland", "Murray State", 0.976),  # -50000 / +4000
    ("Ole Miss", "Gonzaga", 0.876),  # -1050 / +675
    ("Minnesota", "Green Bay", 0.968),  # -10000 / +3000
    ("Michigan State", "Colorado State", 0.940),  # -3600 / +1500
    ("NC State", "Tennessee", 0.532),  # -125 / +105 (NC State fav!)
    ("Texas Tech", "Villanova", 0.551),  # -135 / +114
    ("Oklahoma", "Idaho", 0.990),  # off (spread -34.5)
    ("Notre Dame", "Fairfield", 0.787),  # -455 / +350
    ("Kentucky", "James Madison", 0.926),  # -2400 / +1200
    ("Alabama", "Rhode Island", 0.782),  # -440 / +340
    ("USC", "Clemson", 0.685),  # -250 / +205
    ("Iowa State", "Syracuse", 0.741),  # -340 / +270
    ("Oklahoma State", "Princeton", 0.675),  # -238 / +195
    ("Illinois", "Colorado", 0.609),  # -175 / +145
    ("Louisville", "Vermont", 0.981),  # -100000 / +5000
    ("Iowa", "Fairleigh Dickinson", 0.981),  # -100000 / +5000
    ("West Virginia", "Miami (OH)", 0.976),  # -50000 / +4000
    # Extreme blowouts (moneyline off, estimate from spread)
    ("Duke", "Charleston", 0.995),  # spread -31.5
    ("TCU", "UC San Diego", 0.995),  # spread -34.5
    ("Michigan", "Holy Cross", 0.998),  # spread -41.5
    ("North Carolina", "Western Illinois", 0.985),  # spread -25.5
    ("LSU", "Jacksonville", 0.998),  # spread -51.5
    ("Ohio State", "Howard", 0.995),  # spread -37.5
    ("UConn", "UTSA", 0.999),  # spread -54.5
    ("Vanderbilt", "High Point", 0.995),  # spread -36.5
    ("UCLA", "California Baptist", 0.998),  # spread -51.5
]


def _build_vegas_w_r1_games() -> dict[tuple[int, int], float]:
    """Convert W Vegas R1 predictions to {(min_id, max_id): P(min_id wins)}."""
    games: dict[tuple[int, int], float] = {}
    for fav, dog, fav_prob in _VEGAS_W_R1_RAW:
        fav_id = BPI_W_TEAM_MAP.get(fav)
        dog_id = BPI_W_TEAM_MAP.get(dog)
        if fav_id is None or dog_id is None:
            print(
                f"  WARNING: unmapped W Vegas team: {fav} ({fav_id}) or {dog} ({dog_id})"
            )
            continue
        t1 = min(fav_id, dog_id)
        t2 = max(fav_id, dog_id)
        if t1 == fav_id:
            games[(t1, t2)] = fav_prob
        else:
            games[(t1, t2)] = 1.0 - fav_prob
    return games


VEGAS_W_R1_GAMES = _build_vegas_w_r1_games()


# ── Market consensus: 60% Vegas / 40% BPI (Vegas reacts faster to injuries) ──

VEGAS_WEIGHT = 0.60  # Vegas has real money → more efficient


def market_consensus(
    t1: int,
    t2: int,
    bpi_games: dict[tuple[int, int], float],
    vegas_games: dict[tuple[int, int], float],
) -> float | None:
    """Weighted consensus of BPI and Vegas. Returns None if neither available."""
    key = (t1, t2)
    bpi = bpi_games.get(key)
    vegas = vegas_games.get(key)
    if bpi is not None and vegas is not None:
        return VEGAS_WEIGHT * vegas + (1 - VEGAS_WEIGHT) * bpi
    if bpi is not None:
        return bpi
    if vegas is not None:
        return vegas
    return None


# ── Blend alphas ──────────────────────────────────────────────────

# M blend — aggressive market trust for R1 (market knows injuries/travel)
ALPHA_M_R1 = 0.80  # M R1: 80% market_consensus (avg BPI+Vegas)
ALPHA_M_BPI_BT = 0.25  # M non-R1: 25% BPI Bradley-Terry
ALPHA_M_KALSHI = 0.15  # M fallback: 15% Kalshi BT

# W blend — market has better game-specific info
ALPHA_W_R1 = 0.65  # W R1: 65% BPI game-specific
ALPHA_W_BPI_BT = 0.15  # W non-R1: 15% BPI BT
ALPHA_W_KALSHI = 0.15  # W fallback: 15% Kalshi BT


# ── CV evaluation (for validation) ───────────────────────────────


def cv_lr(
    gender: str,
    feats: list[str],
    C: float,
    clip_low: float,
    clip_high: float,
) -> float:
    """LOSO CV. Returns Brier score."""
    tourney = _m_tourney if gender == "M" else _w_tourney
    _build = build_m_features if gender == "M" else build_w_features

    rows: list[dict[str, float]] = []
    for season in TRAIN_SEASONS:
        games = tourney[tourney["Season"] == season]
        if len(games) == 0:
            continue
        features = _build(season)
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
                ("lr", LogisticRegression(C=C, max_iter=1000, solver="lbfgs")),
            ]
        )
        pipe.fit(X[train_idx], y[train_idx])
        oof[val_idx] = pipe.predict_proba(X[val_idx])[:, 1]
    oof = np.clip(oof, clip_low, clip_high)
    return float(brier_score_loss(y, oof))


# ── Generate submission ────────────────────────────────────────────


def generate_submission(tag: str = "r52_best") -> Path:
    """Generate submission with triple-market blend."""
    m_odds, w_odds = load_kalshi_odds()

    print("  Market data summary:")
    print(f"    M BPI R1: {len(BPI_R1_GAMES)} games")
    print(f"    M Vegas R1: {len(VEGAS_M_R1_GAMES)} games")
    print(f"    M BPI champ: {len(BPI_M_CHAMP)} teams (BT pairwise)")
    print(f"    W BPI R1: {len(BPI_W_R1_GAMES)} games")
    print(f"    W Vegas R1: {len(VEGAS_W_R1_GAMES)} games")
    print(f"    W BPI champ: {len(BPI_W_CHAMP)} teams (BT pairwise)")
    print(f"    Kalshi: {len(m_odds)} M, {len(w_odds)} W")

    # Check overlap between BPI and Vegas R1
    overlap = set(BPI_R1_GAMES.keys()) & set(VEGAS_M_R1_GAMES.keys())
    bpi_only = set(BPI_R1_GAMES.keys()) - set(VEGAS_M_R1_GAMES.keys())
    vegas_only = set(VEGAS_M_R1_GAMES.keys()) - set(BPI_R1_GAMES.keys())
    print(
        f"    R1 overlap: {len(overlap)} games, BPI-only: {len(bpi_only)}, Vegas-only: {len(vegas_only)}"
    )

    # Show BPI vs Vegas comparison for closest games
    print("\n  BPI vs Vegas comparison (closest games):")
    combined = []
    for key in sorted(overlap):
        bpi_p = BPI_R1_GAMES[key]
        vegas_p = VEGAS_M_R1_GAMES[key]
        consensus = (bpi_p + vegas_p) / 2.0
        diff = abs(bpi_p - vegas_p)
        combined.append((key, bpi_p, vegas_p, consensus, diff))
    combined.sort(key=lambda x: -x[4])  # sort by biggest disagreement
    for key, bpi_p, vegas_p, cons, diff in combined[:10]:
        print(
            f"    {key}: BPI={bpi_p:.3f}, Vegas={vegas_p:.3f}, Consensus={cons:.3f} (Δ={diff:.3f})"
        )

    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds: dict[str, float] = {}
    stats: dict[str, int] = {
        "m_r1_consensus": 0,
        "m_r1_bpi_only": 0,
        "m_r1_vegas_only": 0,
        "m_bpi_bt": 0,
        "m_kalshi": 0,
        "m_pure_lr": 0,
        "w_r1_consensus": 0,
        "w_bpi_bt": 0,
        "w_kalshi": 0,
        "w_pure_lr": 0,
    }

    for gender, feats, c_val, clip_lo, clip_hi in [
        ("M", list(FEATURES_M), C_M, CLIP_M_LOW, CLIP_M_HIGH),
        ("W", list(FEATURES_W), C_W, CLIP_W_LOW, CLIP_W_HIGH),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney
        build_fn = build_m_features if gender == "M" else build_w_features

        # Build full training data (all seasons)
        rows: list[dict[str, float]] = []
        for season in TRAIN_SEASONS:
            games = tourney[tourney["Season"] == season]
            if len(games) == 0:
                continue
            features = build_fn(season)
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
                ("lr", LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")),
            ]
        )
        pipe.fit(X_train, y_train)

        # Predict 2026
        features_2026 = build_fn(2026)

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
            final = lr_pred
            key = (t1, t2)

            if gender == "M":
                # Tier 1: R1 game-specific (triple-market consensus)
                cons = market_consensus(t1, t2, BPI_R1_GAMES, VEGAS_M_R1_GAMES)
                if cons is not None:
                    final = (1 - ALPHA_M_R1) * lr_pred + ALPHA_M_R1 * cons
                    # Track which source
                    if key in BPI_R1_GAMES and key in VEGAS_M_R1_GAMES:
                        stats["m_r1_consensus"] += 1
                    elif key in BPI_R1_GAMES:
                        stats["m_r1_bpi_only"] += 1
                    else:
                        stats["m_r1_vegas_only"] += 1
                # Tier 2: BPI Bradley-Terry (championship strength)
                elif t1 in BPI_M_CHAMP and t2 in BPI_M_CHAMP:
                    mkt = bpi_bt_pairwise(t1, t2)
                    final = (1 - ALPHA_M_BPI_BT) * lr_pred + ALPHA_M_BPI_BT * mkt
                    stats["m_bpi_bt"] += 1
                # Tier 3: Kalshi Bradley-Terry
                elif t1 in m_odds and t2 in m_odds:
                    mkt = kalshi_pairwise(t1, t2, m_odds)
                    final = (1 - ALPHA_M_KALSHI) * lr_pred + ALPHA_M_KALSHI * mkt
                    stats["m_kalshi"] += 1
                # Tier 4: Pure LR
                else:
                    stats["m_pure_lr"] += 1

            else:  # Women's
                # Tier 1: R1 game-specific (BPI + Vegas consensus)
                cons = market_consensus(t1, t2, BPI_W_R1_GAMES, VEGAS_W_R1_GAMES)
                if cons is not None:
                    final = (1 - ALPHA_W_R1) * lr_pred + ALPHA_W_R1 * cons
                    stats["w_r1_consensus"] += 1
                # Tier 2: BPI Bradley-Terry
                elif t1 in BPI_W_CHAMP and t2 in BPI_W_CHAMP:
                    market = bpi_w_bt_pairwise(t1, t2)
                    final = (1 - ALPHA_W_BPI_BT) * lr_pred + ALPHA_W_BPI_BT * market
                    stats["w_bpi_bt"] += 1
                # Tier 3: Kalshi Bradley-Terry
                elif t1 in w_odds and t2 in w_odds:
                    market = kalshi_pairwise(t1, t2, w_odds)
                    final = (1 - ALPHA_W_KALSHI) * lr_pred + ALPHA_W_KALSHI * market
                    stats["w_kalshi"] += 1
                # Tier 4: Pure LR
                else:
                    stats["w_pure_lr"] += 1

            preds[game_id] = float(np.clip(final, clip_lo, clip_hi))

    sub["Pred"] = sub["ID"].map(preds).fillna(0.5)
    SUBMISSIONS.mkdir(exist_ok=True)
    out_path = SUBMISSIONS / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)

    print(f"\n  Submission: {out_path}")
    print(f"  Predictions: {len(preds)}")
    print(f"  Range: [{sub['Pred'].min():.4f}, {sub['Pred'].max():.4f}]")
    print(f"  Mean: {sub['Pred'].mean():.4f}")
    print("  Blend stats:")
    print(
        f"    M: R1_consensus={stats['m_r1_consensus']}, R1_BPI_only={stats['m_r1_bpi_only']}, "
        f"R1_Vegas_only={stats['m_r1_vegas_only']}, BPI_BT={stats['m_bpi_bt']}, "
        f"Kalshi={stats['m_kalshi']}, Pure={stats['m_pure_lr']}"
    )
    print(
        f"    W: R1_consensus={stats['w_r1_consensus']}, BPI_BT={stats['w_bpi_bt']}, "
        f"Kalshi={stats['w_kalshi']}, Pure={stats['w_pure_lr']}"
    )

    # Sanity checks — key M games with injury impact
    print("\n  M R1 sanity checks (injury-impacted games):")
    m_checks = [
        ("UNC vs VCU", "2026_1314_1433", "Wilson out (19.8 PPG)"),
        ("BYU vs Texas", "2026_1140_1400", "Saunders ACL"),
        ("Louisville vs USF", "2026_1257_1378", "Mikel Brown out"),
        ("Alabama vs Hofstra", "2026_1104_1220", "Holloway arrested"),
        ("Ohio State vs TCU", "2026_1326_1395", "8v9 toss-up"),
        ("Georgia vs Saint Louis", "2026_1208_1387", "8v9 toss-up"),
        ("Utah State vs Villanova", "2026_1429_1437", "8v9 toss-up"),
        ("Iowa vs Clemson", "2026_1155_1234", "8v9 toss-up"),
        ("Miami vs Missouri", "2026_1274_1281", "7v10 toss-up"),
        ("Duke vs Siena", "2026_1181_1373", "1v16 chalk"),
    ]
    for label, gid, note in m_checks:
        if gid in preds:
            # Show LR vs market breakdown
            parts = gid.split("_")
            t1, t2 = int(parts[1]), int(parts[2])
            bpi_val = BPI_R1_GAMES.get((t1, t2), None)
            vegas_val = VEGAS_M_R1_GAMES.get((t1, t2), None)
            consensus_val = market_consensus(t1, t2, BPI_R1_GAMES, VEGAS_M_R1_GAMES)
            bpi_s = f"{bpi_val:.3f}" if bpi_val is not None else "  N/A"
            vegas_s = f"{vegas_val:.3f}" if vegas_val is not None else "  N/A"
            cons_s = f"{consensus_val:.3f}" if consensus_val is not None else "  N/A"
            print(
                f"    {label}: final={preds[gid]:.4f} | "
                f"BPI={bpi_s} | Vegas={vegas_s} | Consensus={cons_s} | {note}"
            )

    # W sanity checks
    print("\n  W R1 sanity checks (BPI vs Vegas):")
    w_checks = [
        ("UConn vs UTSA", "2026_3163_3427", "1-seed dominant"),
        ("Tennessee vs NC State", "2026_3301_3397", "BPI↔Vegas disagree!"),
        ("Texas Tech vs Villanova", "2026_3403_3437", "Close game"),
        ("Iowa State vs Syracuse", "2026_3235_3393", "Close game"),
        ("Oregon vs Virginia Tech", "2026_3332_3439", "Close game"),
    ]
    for label, gid, note in w_checks:
        if gid in preds:
            parts_w = gid.split("_")
            t1_w, t2_w = int(parts_w[1]), int(parts_w[2])
            bpi_w = BPI_W_R1_GAMES.get((t1_w, t2_w))
            vegas_w = VEGAS_W_R1_GAMES.get((t1_w, t2_w))
            cons_w = market_consensus(t1_w, t2_w, BPI_W_R1_GAMES, VEGAS_W_R1_GAMES)
            bpi_ws = f"{bpi_w:.3f}" if bpi_w is not None else "  N/A"
            vegas_ws = f"{vegas_w:.3f}" if vegas_w is not None else "  N/A"
            cons_ws = f"{cons_w:.3f}" if cons_w is not None else "  N/A"
            print(
                f"    {label}: final={preds[gid]:.4f} | "
                f"BPI={bpi_ws} | Vegas={vegas_ws} | Cons={cons_ws} | {note}"
            )

    return out_path


# Also generate a pure-LR submission for comparison
def generate_pure_lr(tag: str = "r52_pure") -> Path:
    """Generate pure LR submission (no market blend) for comparison."""
    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds: dict[str, float] = {}

    for gender, feats, c_val, clip_lo, clip_hi in [
        ("M", list(FEATURES_M), C_M, CLIP_M_LOW, CLIP_M_HIGH),
        ("W", list(FEATURES_W), C_W, CLIP_W_LOW, CLIP_W_HIGH),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney
        build_fn = build_m_features if gender == "M" else build_w_features

        rows: list[dict[str, float]] = []
        for season in TRAIN_SEASONS:
            games = tourney[tourney["Season"] == season]
            if len(games) == 0:
                continue
            features = build_fn(season)
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
                ("lr", LogisticRegression(C=c_val, max_iter=1000, solver="lbfgs")),
            ]
        )
        pipe.fit(X_train, y_train)

        features_2026 = build_fn(2026)

        for _, row_sub in sub.iterrows():
            game_id = row_sub["ID"]
            parts = game_id.split("_")
            if len(parts) != 3 or int(parts[0]) != 2026:
                continue
            t1, t2 = int(parts[1]), int(parts[2])
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
            lr_pred = float(pipe.predict_proba(x_arr)[:, 1][0])
            preds[game_id] = float(np.clip(lr_pred, clip_lo, clip_hi))

    sub["Pred"] = sub["ID"].map(preds).fillna(0.5)
    SUBMISSIONS.mkdir(exist_ok=True)
    out_path = SUBMISSIONS / f"submission_{tag}.csv"
    sub.to_csv(out_path, index=False)
    print(f"\n  Pure LR: {out_path}")
    return out_path


# ── Main ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("March Mania 2026 — Round 52: Triple-Market Blend")
    print(f"  Base: R50 features (M={len(FEATURES_M)}, W={len(FEATURES_W)})")
    print(
        f"  Market: BPI ({len(BPI_R1_GAMES)} M R1) + Vegas ({len(VEGAS_M_R1_GAMES)} M R1)"
    )
    print(
        f"  W Market: BPI ({len(BPI_W_R1_GAMES)} R1) + Vegas ({len(VEGAS_W_R1_GAMES)} R1)"
    )
    print(f"  Consensus: {VEGAS_WEIGHT:.0%} Vegas + {1 - VEGAS_WEIGHT:.0%} BPI")
    print(f"  Blend: M R1={ALPHA_M_R1:.0%} market, W R1={ALPHA_W_R1:.0%} market")
    print("=" * 70)

    # Confirm R50 CV baseline
    print("\n── CV Baseline (R50 features, no market) ──")
    m_cv = cv_lr("M", list(FEATURES_M), C_M, CLIP_M_LOW, CLIP_M_HIGH)
    w_cv = cv_lr("W", list(FEATURES_W), C_W, CLIP_W_LOW, CLIP_W_HIGH)
    print(f"  M={m_cv:.5f}, W={w_cv:.5f}, Combined={0.5 * (m_cv + w_cv):.5f}")

    # Generate submissions
    print("\n── Generating triple-market blend submission ──")
    best_path = generate_submission(tag="r52_best")

    print("\n── Generating pure LR submission (for comparison) ──")
    pure_path = generate_pure_lr(tag="r52_pure")

    # Compare key games between pure and blend
    print("\n── Pure LR vs Triple-Market Blend (key M R1 games) ──")
    pure_df = pd.read_csv(pure_path)
    best_df = pd.read_csv(best_path)
    pure_map = dict(zip(pure_df["ID"], pure_df["Pred"]))
    best_map = dict(zip(best_df["ID"], best_df["Pred"]))

    close_games = [
        ("UNC vs VCU", "2026_1314_1433"),
        ("BYU vs Texas", "2026_1140_1400"),
        ("Ohio State vs TCU", "2026_1326_1395"),
        ("Georgia vs Saint Louis", "2026_1208_1387"),
        ("Utah State vs Villanova", "2026_1429_1437"),
        ("Iowa vs Clemson", "2026_1155_1234"),
        ("Miami vs Missouri", "2026_1274_1281"),
        ("Kentucky vs Santa Clara", "2026_1246_1365"),
        ("Saint Mary's vs Texas A&M", "2026_1388_1401"),
    ]
    for label, gid in close_games:
        pure_p = pure_map.get(gid, 0.5)
        best_p = best_map.get(gid, 0.5)
        diff = best_p - pure_p
        print(f"  {label:30s}: LR={pure_p:.4f} → Blend={best_p:.4f} (Δ={diff:+.4f})")

    print("\n" + "=" * 70)
    print("DONE — Review before submitting!")
    print("=" * 70)
