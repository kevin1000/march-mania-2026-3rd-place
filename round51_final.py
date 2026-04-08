"""
March Machine Learning Mania 2026 — Round 51: Final Push

R50 CV: 0.12524 (M=0.11541, W=0.13506)
R51 improvements over R50:
  - M: +Qual D (quality-opponent adjusted defense, r=0.036 with adjde)
  - M: +Qual O (quality-opponent adjusted offense, test)
  - M: +Coach PASE (Performance Above Seed Expectation)
  - W: +ESPN BPI R1 game-specific predictions (replaces thin Kalshi data)
  - C and clip re-tuning after feature changes

Market blend:
  - M R1: 50% BPI game-specific + 50% LR
  - M other: 35% BPI BT + 65% LR
  - W R1: 40% BPI game-specific + 60% LR (NEW)
  - W other: 20% BPI BT + 80% LR (NEW, replaces 15% Kalshi)
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
    build_m_features as _build_m_r50,
    build_w_features,
    FEATURES_M as _FEATURES_M_R50,
    FEATURES_W,
    C_W,
    CLIP_W_LOW,
    CLIP_W_HIGH,
    BPI_M_CHAMP,
    BPI_R1_GAMES,
    bpi_bt_pairwise,
)
from round46_final import (
    load_kalshi_odds,
    kalshi_pairwise,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
EXT = HERE / "external"
SUBMISSIONS = HERE / "submissions"

TRAIN_SEASONS = [y for y in range(2015, 2026) if y != 2020]

# ── R51 M features: R50 + Qual D, Qual O, Coach PASE (test each) ──

FEATURES_M_BASE = list(_FEATURES_M_R50)  # 24 features from R50

# Start with R50 params, will re-tune after feature experiments
C_M = 500.0
CLIP_M_LOW, CLIP_M_HIGH = 0.03, 0.97


# ── Coach PASE feature ──────────────────────────────────────────


def _load_coach_pase() -> dict[str, float]:
    """Load coach name → PASE from Coach Results.csv."""
    cr = pd.read_csv(EXT / "Coach Results.csv")
    result: dict[str, float] = {}
    for _, row in cr.iterrows():
        name = str(row["COACH"]).strip().lower().replace(" ", "_")
        name = name.replace(".", "")
        result[name] = float(row["PASE"])
    return result


_coach_pase_map = _load_coach_pase()


def _get_coach_pase(season: int) -> pd.DataFrame:
    """Get coach PASE for each team in a season."""
    coaches = pd.read_csv(DATA / "MTeamCoaches.csv")
    season_coaches = coaches[coaches["Season"] == season].copy()
    # Keep the last (most recent) coach for each team
    season_coaches = season_coaches.sort_values("LastDayNum").drop_duplicates(
        subset=["TeamID"], keep="last"
    )
    season_coaches["coach_pase"] = season_coaches["CoachName"].map(_coach_pase_map)
    return season_coaches[["TeamID", "coach_pase"]].set_index("TeamID")


def build_m_features(season: int) -> pd.DataFrame:
    """R50 M features + Qual D, Qual O, Coach PASE."""
    features = _build_m_r50(season)
    # Qual D and Qual O are now extracted by _get_barttorvik (added to round27_pruned)
    # They should already be in the features DataFrame via the build chain
    # If not present (column names may differ), add them explicitly
    if "Qual D" not in features.columns or "Qual O" not in features.columns:
        from round27_pruned import read_barttorvik, _bt_map

        filepath = EXT / (
            "barttorvik_2026_live.csv" if season == 2026 else f"barttorvik_{season}.csv"
        )
        if filepath.exists():
            bt = read_barttorvik(filepath)
            bt = bt.merge(_bt_map, on="team", how="inner")
            bt = bt.drop_duplicates(subset=["TeamID"], keep="first").set_index("TeamID")
            for col in ["Qual D", "Qual O"]:
                if col in bt.columns:
                    features[col] = bt[col].reindex(features.index)
                    features[col] = pd.to_numeric(features[col], errors="coerce")

    # Coach PASE
    if "coach_pase" not in features.columns:
        coach_df = _get_coach_pase(season)
        features = features.join(coach_df, how="left")

    return features


# ── ESPN Women's BPI Data (scraped March 19, 2026) ───────────────

BPI_W_TEAM_MAP: dict[str, int] = {
    "Charleston": 3158,
    "UC San Diego": 3471,
    "TCU": 3395,
    "Virginia Tech": 3439,
    "Oregon": 3332,
    "South Dakota State": 3355,
    "Washington": 3449,
    "Murray State": 3293,
    "Maryland": 3268,
    "Gonzaga": 3211,
    "Ole Miss": 3279,
    "Holy Cross": 3221,
    "Western Illinois": 3442,
    "Michigan": 3276,
    "North Carolina": 3314,
    "Jacksonville": 3239,
    "LSU": 3261,
    "Green Bay": 3453,
    "Minnesota": 3278,
    "Colorado State": 3161,
    "Michigan State": 3277,
    "Tennessee": 3397,
    "NC State": 3301,
    "Villanova": 3437,
    "Texas Tech": 3403,
    "Idaho": 3225,
    "Oklahoma": 3328,
    "Howard": 3224,
    "Ohio State": 3326,
    "Vermont": 3436,
    "Louisville": 3257,
    "South Carolina": 3376,
    "Georgia": 3208,
    "Fairfield": 3193,
    "Notre Dame": 3323,
    "James Madison": 3241,
    "Kentucky": 3246,
    "Rhode Island": 3348,
    "Alabama": 3104,
    "UConn": 3163,
    "USC": 3425,
    "Clemson": 3155,
    "Fairleigh Dickinson": 3192,
    "Iowa": 3234,
    "Miami (OH)": 3275,
    "West Virginia": 3452,
    "Syracuse": 3393,
    "Iowa State": 3235,
    "High Point": 3219,
    "Vanderbilt": 3435,
    "Princeton": 3343,
    "Oklahoma State": 3329,
    "Colorado": 3160,
    "Illinois": 3228,
    "California Baptist": 3465,
    "UCLA": 3417,
    "UTSA": 3427,
    "Duke": 3181,
    "Nebraska": 3304,
    "Richmond": 3350,
    "Stephen F. Austin": 3372,
    "Missouri State": 3283,
    "Samford": 3359,
    "Southern": 3380,
    "Arizona State": 3113,
    "Virginia": 3438,
    "Baylor": 3124,
    "Texas": 3400,
}

# W BPI championship-level strength (for BT pairwise on non-R1 games)
# Use top teams from ESPN BPI + Kalshi, rest get floor
BPI_W_CHAMP: dict[int, float] = {
    3163: 30.0,  # UConn (dominant)
    3417: 10.0,  # UCLA
    3376: 6.0,  # South Carolina
    3400: 5.5,  # Texas
    3261: 3.0,  # LSU
    3181: 2.0,  # Duke
    3314: 1.5,  # North Carolina
    3323: 1.5,  # Notre Dame
    3235: 1.0,  # Iowa State
    3234: 1.0,  # Iowa
    3124: 0.8,  # Baylor
    3301: 0.5,  # NC State
    3246: 0.5,  # Kentucky
    3257: 0.5,  # Louisville
    3425: 0.5,  # USC
    3395: 0.5,  # TCU
    3397: 0.5,  # Tennessee
    3328: 0.4,  # Oklahoma
    3435: 0.4,  # Vanderbilt
    3276: 0.3,  # Michigan
    3279: 0.3,  # Ole Miss
    3348: 0.2,  # Rhode Island
    3268: 0.2,  # Maryland
    3277: 0.2,  # Michigan State
    3326: 0.2,  # Ohio State
    3332: 0.2,  # Oregon
    3104: 0.2,  # Alabama
    3228: 0.1,  # Illinois
    3452: 0.1,  # West Virginia
    3278: 0.1,  # Minnesota
}

# W R1 game-specific BPI predictions
_BPI_W_R1_RAW: list[tuple[str, str, float]] = [
    # (favored, underdog, favored_win_prob)
    ("Duke", "Charleston", 0.982),
    ("TCU", "UC San Diego", 0.985),
    ("Oregon", "Virginia Tech", 0.655),
    ("Washington", "South Dakota State", 0.661),
    ("Maryland", "Murray State", 0.943),
    ("Ole Miss", "Gonzaga", 0.865),
    ("Michigan", "Holy Cross", 0.994),
    ("North Carolina", "Western Illinois", 0.976),
    ("LSU", "Jacksonville", 0.998),
    ("Minnesota", "Green Bay", 0.976),
    ("Michigan State", "Colorado State", 0.927),
    ("Tennessee", "NC State", 0.555),
    ("Texas Tech", "Villanova", 0.522),
    ("Oklahoma", "Idaho", 0.981),
    ("Ohio State", "Howard", 0.991),
    ("Louisville", "Vermont", 0.976),
    ("Notre Dame", "Fairfield", 0.860),
    ("Kentucky", "James Madison", 0.865),
    ("Alabama", "Rhode Island", 0.819),
    ("UConn", "UTSA", 0.999),
    ("USC", "Clemson", 0.761),
    ("Iowa", "Fairleigh Dickinson", 0.980),
    ("West Virginia", "Miami (OH)", 0.962),
    ("Iowa State", "Syracuse", 0.637),
    ("Vanderbilt", "High Point", 0.988),
    ("Oklahoma State", "Princeton", 0.617),
    ("Illinois", "Colorado", 0.686),
    ("UCLA", "California Baptist", 0.998),
]


def _build_bpi_w_r1_games() -> dict[tuple[int, int], float]:
    """Convert W BPI R1 predictions to {(min_id, max_id): P(min_id wins)} format."""
    games: dict[tuple[int, int], float] = {}
    for fav, dog, fav_prob in _BPI_W_R1_RAW:
        fav_id = BPI_W_TEAM_MAP.get(fav)
        dog_id = BPI_W_TEAM_MAP.get(dog)
        if fav_id is None or dog_id is None:
            print(f"  WARNING: unmapped W R1 team: {fav} or {dog}")
            continue
        t1 = min(fav_id, dog_id)
        t2 = max(fav_id, dog_id)
        if t1 == fav_id:
            games[(t1, t2)] = fav_prob
        else:
            games[(t1, t2)] = 1.0 - fav_prob
    return games


BPI_W_R1_GAMES = _build_bpi_w_r1_games()


def bpi_w_bt_pairwise(t1: int, t2: int, floor: float = 0.05) -> float:
    """Bradley-Terry from W BPI championship estimates."""
    s1 = BPI_W_CHAMP.get(t1, floor)
    s2 = BPI_W_CHAMP.get(t2, floor)
    return s1 / (s1 + s2)


# ── Blend alphas ──────────────────────────────────────────────────

ALPHA_M_R1 = 0.50  # M R1: game-specific BPI (unchanged from R50)
ALPHA_M_BPI_BT = 0.35  # M other: BPI BT (unchanged)
ALPHA_M_KALSHI = 0.20  # M Kalshi fallback (unchanged)
ALPHA_W_R1 = 0.40  # W R1: game-specific BPI (NEW)
ALPHA_W_BPI_BT = 0.20  # W other: BPI BT (NEW)
ALPHA_W_KALSHI = 0.15  # W Kalshi fallback (unchanged)


# ── CV evaluation ──────────────────────────────────────────────────


def cv_lr(
    gender: str,
    feats: list[str],
    C: float,
    clip_low: float,
    clip_high: float,
    build_fn: callable | None = None,
) -> float:
    """LOSO CV. Returns Brier score."""
    tourney = _m_tourney if gender == "M" else _w_tourney
    _build = build_fn or (build_m_features if gender == "M" else build_w_features)

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


def run_experiments() -> tuple[list[str], float]:
    """Test new features, return best M feature list and Brier."""
    print("\n── Feature Experiments ──")

    # R50 baseline
    base_m = cv_lr("M", FEATURES_M_BASE, C_M, CLIP_M_LOW, CLIP_M_HIGH)
    base_w = cv_lr("W", list(FEATURES_W), C_W, CLIP_W_LOW, CLIP_W_HIGH)
    print(
        f"  R50 baseline: M={base_m:.5f}, W={base_w:.5f}, Combined={0.5 * (base_m + base_w):.5f}"
    )

    # Test adding Qual D to M
    m_feats = list(FEATURES_M_BASE)
    m_with_qd = m_feats + ["Qual D"]
    b = cv_lr("M", m_with_qd, C_M, CLIP_M_LOW, CLIP_M_HIGH)
    d = b - base_m
    print(f"  M +Qual D: {b:.5f} (Δ={d:+.5f}) {'<-- BETTER' if d < 0 else ''}")
    if d < 0:
        m_feats = m_with_qd
        base_m = b

    # Test adding Qual O to M
    m_with_qo = m_feats + ["Qual O"]
    b = cv_lr("M", m_with_qo, C_M, CLIP_M_LOW, CLIP_M_HIGH)
    d = b - base_m
    print(f"  M +Qual O: {b:.5f} (Δ={d:+.5f}) {'<-- BETTER' if d < 0 else ''}")
    if d < 0:
        m_feats = m_with_qo
        base_m = b

    # Test adding Coach PASE to M
    m_with_cp = m_feats + ["coach_pase"]
    b = cv_lr("M", m_with_cp, C_M, CLIP_M_LOW, CLIP_M_HIGH)
    d = b - base_m
    print(f"  M +coach_pase: {b:.5f} (Δ={d:+.5f}) {'<-- BETTER' if d < 0 else ''}")
    if d < 0:
        m_feats = m_with_cp
        base_m = b

    # C re-tune for M
    print(f"\n  M C sweep (current C={C_M}, {len(m_feats)} feats):")
    best_c_m, best_b_m = C_M, base_m
    for c in [30, 50, 75, 100, 150, 200, 300, 500, 750, 1000]:
        b = cv_lr("M", m_feats, float(c), CLIP_M_LOW, CLIP_M_HIGH)
        marker = " <-- BEST" if b < best_b_m else ""
        print(f"    C={c:>6}: {b:.5f} (Δ={b - base_m:+.5f}){marker}")
        if b < best_b_m:
            best_c_m, best_b_m = float(c), b

    combined = 0.5 * (best_b_m + base_w)
    print(f"\n  RESULT: M={best_b_m:.5f} (C={best_c_m}), W={base_w:.5f}")
    print(f"  Combined: {combined:.5f}")
    print(f"  M features ({len(m_feats)}): {m_feats}")
    return m_feats, best_c_m


# ── Generate submission ────────────────────────────────────────────


def generate_submission(
    m_feats: list[str],
    m_c: float,
    tag: str = "r51_best",
) -> Path:
    """Generate single best submission with full market blend."""
    m_odds, w_odds = load_kalshi_odds()
    print("  Market data:")
    print(f"    M BPI: {len(BPI_M_CHAMP)} champ, {len(BPI_R1_GAMES)} R1 games")
    print(f"    W BPI: {len(BPI_W_CHAMP)} champ, {len(BPI_W_R1_GAMES)} R1 games")
    print(f"    Kalshi: {len(m_odds)} M, {len(w_odds)} W")

    sub = pd.read_csv(DATA / "SampleSubmissionStage2.csv")
    preds: dict[str, float] = {}
    stats: dict[str, int] = {
        "m_r1_bpi": 0,
        "m_bpi_bt": 0,
        "m_kalshi": 0,
        "m_pure_lr": 0,
        "w_r1_bpi": 0,
        "w_bpi_bt": 0,
        "w_kalshi": 0,
        "w_pure_lr": 0,
    }

    for gender, feats, c_val, clip_lo, clip_hi in [
        ("M", m_feats, m_c, CLIP_M_LOW, CLIP_M_HIGH),
        ("W", list(FEATURES_W), C_W, CLIP_W_LOW, CLIP_W_HIGH),
    ]:
        tourney = _m_tourney if gender == "M" else _w_tourney
        build_fn = build_m_features if gender == "M" else build_w_features

        # Build training data
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
                if key in BPI_R1_GAMES:
                    market = BPI_R1_GAMES[key]
                    final = (1 - ALPHA_M_R1) * lr_pred + ALPHA_M_R1 * market
                    stats["m_r1_bpi"] += 1
                elif t1 in BPI_M_CHAMP and t2 in BPI_M_CHAMP:
                    market = bpi_bt_pairwise(t1, t2)
                    final = (1 - ALPHA_M_BPI_BT) * lr_pred + ALPHA_M_BPI_BT * market
                    stats["m_bpi_bt"] += 1
                elif t1 in m_odds and t2 in m_odds:
                    market = kalshi_pairwise(t1, t2, m_odds)
                    final = (1 - ALPHA_M_KALSHI) * lr_pred + ALPHA_M_KALSHI * market
                    stats["m_kalshi"] += 1
                else:
                    stats["m_pure_lr"] += 1

            else:  # Women's
                if key in BPI_W_R1_GAMES:
                    market = BPI_W_R1_GAMES[key]
                    final = (1 - ALPHA_W_R1) * lr_pred + ALPHA_W_R1 * market
                    stats["w_r1_bpi"] += 1
                elif t1 in BPI_W_CHAMP and t2 in BPI_W_CHAMP:
                    market = bpi_w_bt_pairwise(t1, t2)
                    final = (1 - ALPHA_W_BPI_BT) * lr_pred + ALPHA_W_BPI_BT * market
                    stats["w_bpi_bt"] += 1
                elif t1 in w_odds and t2 in w_odds:
                    market = kalshi_pairwise(t1, t2, w_odds)
                    final = (1 - ALPHA_W_KALSHI) * lr_pred + ALPHA_W_KALSHI * market
                    stats["w_kalshi"] += 1
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
        f"    M: R1_BPI={stats['m_r1_bpi']}, BPI_BT={stats['m_bpi_bt']}, "
        f"Kalshi={stats['m_kalshi']}, Pure={stats['m_pure_lr']}"
    )
    print(
        f"    W: R1_BPI={stats['w_r1_bpi']}, BPI_BT={stats['w_bpi_bt']}, "
        f"Kalshi={stats['w_kalshi']}, Pure={stats['w_pure_lr']}"
    )

    # Sanity checks
    print("\n  W R1 sanity checks:")
    w_checks = [
        ("UConn vs UTSA", "2026_3163_3427", 0.999),
        ("Tennessee vs NC State", "2026_3301_3397", 0.445),
        ("Iowa State vs Syracuse", "2026_3235_3393", 0.637),
        ("Oregon vs Virginia Tech", "2026_3332_3439", 0.655),
    ]
    for label, gid, bpi_approx in w_checks:
        if gid in preds:
            print(f"    {label}: pred={preds[gid]:.4f} (BPI~{bpi_approx:.3f})")

    return out_path


# ── Main ──────────────────────────────────────────────────────────


if __name__ == "__main__":
    print("=" * 70)
    print("March Mania 2026 — Round 51: Final Push")
    print("  R50 baseline: M=0.11541, W=0.13506, Combined=0.12524")
    print(f"  New: +Qual D/O, +Coach PASE, +W BPI R1 ({len(BPI_W_R1_GAMES)} games)")
    print("=" * 70)

    m_feats, m_c = run_experiments()

    print("\n── Generating single best submission ──")
    generate_submission(m_feats, m_c, tag="r51_best")

    print("\n" + "=" * 70)
    print("DONE — Review before submitting!")
    print("=" * 70)
