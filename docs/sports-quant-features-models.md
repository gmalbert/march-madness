# Sports-Quant Feature Engineering & Model Architecture

> Source: [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) — March Madness module  
> Purpose: Identify feature engineering techniques and model architecture patterns we can port into our march-madness project.

---

## Executive Summary

The sports-quant repo uses **45 combined features** (21 KenPom difference + 13 BartTorvik difference + 11 matchup interaction) compared to our current **~11 features** (raw efficiency differentials). Key upgrades available:

| Area | Our Current State | sports-quant Approach | Expected Impact |
|------|-------------------|----------------------|-----------------|
| Feature count | ~11 raw diffs | 45 combined difference features | High |
| Feature type | Raw values | Difference features (Team1 − Team2) | High |
| Data symmetrization | None | Double training data via row mirroring | High |
| BartTorvik stats | 2 (AdjOE, AdjDE) | 13 stats including WAB, EliteSOS, QualBarthag | High |
| Matchup interactions | None | 11 cross-source interaction features | Medium |
| Seed priors | None | Historical seed-vs-seed win rates | Medium |

---

## 1. Difference Features (Highest-ROI Change)

Instead of feeding raw team stats as separate columns (Team1_AdjOE, Team2_AdjOE), compute **differences**: `Team1_AdjOE - Team2_AdjOE`. This:
- Cuts feature count in half (36 raw → 21 diff for KenPom alone)
- Eliminates positional bias (the model no longer learns "Team1 usually wins")
- Enables data symmetrization (see Section 4)
- Is the single highest-impact improvement identified in sports-quant's own analysis

### KenPom Difference Features (21 columns)

```python
# From sports-quant _features.py — STAT_PAIRS defines which columns to difference
KENPOM_STAT_PAIRS = [
    ("Rank",        "Rank_Team2"),
    ("AdjEM",       "AdjEM_Team2"),
    ("AdjO",        "AdjO_Team2"),
    ("AdjD",        "AdjD_Team2"),
    ("AdjT",        "AdjT_Team2"),
    ("Luck",        "Luck_Team2"),
    ("AdjEM_SOS",   "AdjEM_SOS_Team2"),
    ("OppO",        "OppO_Team2"),
    ("OppD",        "OppD_Team2"),
    ("AdjEM_NCSOS", "AdjEM_NCSOS_Team2"),
    ("Seed1",       "Seed2"),
    # Additional derived
    ("W_pct",       "W_pct_Team2"),
    ("AdjEM_x_SOS", "AdjEM_x_SOS_Team2"),
    ("Off_Def_ratio", "Off_Def_ratio_Team2"),
    ("Tempo_deviation", "Tempo_deviation_Team2"),
    ("Elite_SOS_proxy", "Elite_SOS_proxy_Team2"),
    ("Consistency_proxy", "Consistency_proxy_Team2"),
    ("SOS_quality_gap", "SOS_quality_gap_Team2"),
    ("Effective_margin", "Effective_margin_Team2"),
    ("Defensive_dominance", "Defensive_dominance_Team2"),
    ("Tournament_readiness", "Tournament_readiness_Team2"),
]

DIFF_FEATURE_COLUMNS = [
    "diff_Rank", "diff_AdjEM", "diff_AdjO", "diff_AdjD", "diff_AdjT",
    "diff_Luck", "diff_AdjEM_SOS", "diff_OppO", "diff_OppD",
    "diff_AdjEM_NCSOS", "diff_Seed",
    "diff_W_pct", "diff_AdjEM_x_SOS", "diff_Off_Def_ratio",
    "diff_Tempo_deviation", "diff_Elite_SOS_proxy",
    "diff_Consistency_proxy", "diff_SOS_quality_gap",
    "diff_Effective_margin", "diff_Defensive_dominance",
    "diff_Tournament_readiness",
]
```

### Implementation for Our Codebase

```python
# feature_engineering_v2.py — Difference feature computation

import pandas as pd
import numpy as np

def compute_difference_features(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw team-pair stats into difference features.
    
    For each (col_team1, col_team2) pair, compute col_team1 - col_team2.
    This eliminates positional bias and halves the feature space.
    
    Args:
        matchups_df: DataFrame with Team1 and Team2 stats in separate columns.
        
    Returns:
        DataFrame with diff_* columns only (no raw team columns).
    """
    STAT_PAIRS = [
        # (Team1_column, Team2_column, output_name)
        # KenPom stats
        ("Rk", "Rk_opp", "diff_Rank"),
        ("NetRtg", "NetRtg_opp", "diff_AdjEM"),
        ("ORtg", "ORtg_opp", "diff_AdjO"),
        ("DRtg", "DRtg_opp", "diff_AdjD"),
        ("AdjT", "AdjT_opp", "diff_AdjT"),
        ("SOS_NetRtg", "SOS_NetRtg_opp", "diff_SOS"),
        # BartTorvik stats
        ("Adj OE", "Adj OE_opp", "diff_Bart_AdjOE"),
        ("Adj DE", "Adj DE_opp", "diff_Bart_AdjDE"),
        ("Barthag", "Barthag_opp", "diff_Barthag"),
        # Seeds
        ("seed", "seed_opp", "diff_Seed"),
    ]
    
    diff_df = pd.DataFrame()
    for col1, col2, out_name in STAT_PAIRS:
        if col1 in matchups_df.columns and col2 in matchups_df.columns:
            diff_df[out_name] = (
                pd.to_numeric(matchups_df[col1], errors="coerce")
                - pd.to_numeric(matchups_df[col2], errors="coerce")
            )
    
    return diff_df
```

---

## 2. BartTorvik Extended Stats (13 columns)

Our repo currently pulls only **Adj OE** and **Adj DE** from BartTorvik. The sports-quant repo pulls **13 columns**, many of which are highly predictive tournament features:

| Column | Description | Why It Matters |
|--------|-------------|----------------|
| `Bart_Rank` | T-Rank overall ranking | Overall quality ranking |
| `Bart_AdjOE` | Adjusted Offensive Efficiency | Points scored per 100 possessions |
| `Bart_AdjDE` | Adjusted Defensive Efficiency | Points allowed per 100 possessions |
| `Bart_Barthag` | Power rating (win prob vs avg team) | Single best predictor of team quality |
| `Bart_AdjT` | Adjusted Tempo | Pace of play |
| `Bart_SOS` | Strength of Schedule | Quality of opponents faced |
| `Bart_NCSOS` | Non-Conference SOS | Schedule strength outside conf play |
| `Bart_EliteSOS` | Elite Strength of Schedule | Quality of top-tier opponents faced |
| `Bart_WAB` | Wins Above Bubble | How many wins above replacement |
| `Bart_QualO` | Quality Offense rating | Offense against quality opponents |
| `Bart_QualD` | Quality Defense rating | Defense against quality opponents |
| `Bart_QualBarthag` | Quality Barthag | Overall quality vs quality opponents |

### Key Stats We're Missing

**WAB (Wins Above Bubble)** — The most tournament-relevant BartTorvik stat. Measures how many wins a team has above what a "bubble team" would achieve with the same schedule. Directly captures tournament-readiness.

**EliteSOS** — Measures performance specifically against elite opponents. Critical for predicting later-round tournament success where matchups are against top teams.

**QualO/QualD/QualBarthag** — Performance ratings filtered to only "quality" opponents (top ~100 teams). Separates teams that beat good opponents from teams that pad stats against weak schedules.

### BartTorvik Time Machine Scraper

The sports-quant repo uses BartTorvik's Time Machine API for **pre-tournament snapshots** — ratings frozen the day before R64 starts. This prevents data leakage from tournament results contaminating predictions.

```python
# barttorvik_time_machine.py — Download pre-tournament snapshots

import gzip
import json
import logging
import urllib.request
import pandas as pd

logger = logging.getLogger(__name__)

# Pre-tournament snapshot dates (day before R64 starts)
SNAPSHOT_DATES = {
    2017: "20170312", 2018: "20180311", 2019: "20190317",
    2021: "20210314", 2022: "20220313", 2023: "20230312",
    2024: "20240317", 2025: "20250316",
}

TIME_MACHINE_URL = (
    "https://barttorvik.com/timemachine/team_results/"
    "{date}_team_results.json.gz"
)

# Column indices in the Time Machine JSON arrays
COL_IDX = {
    "Team": 1, "Bart_Rank": 0, "Bart_AdjOE": 4, "Bart_AdjDE": 6,
    "Bart_Barthag": 8, "Bart_AdjT": 44, "Bart_SOS": 15,
    "Bart_NCSOS": 16, "Bart_EliteSOS": 21, "Bart_WAB": 41,
    "Bart_QualO": 29, "Bart_QualD": 30, "Bart_QualBarthag": 31,
}

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
}


def download_barttorvik_snapshot(year: int) -> pd.DataFrame:
    """Download pre-tournament BartTorvik snapshot from Time Machine.
    
    Uses the day-before-R64 snapshot to prevent tournament data leakage.
    
    Args:
        year: Season year (e.g. 2025 for 2024-25 season).
        
    Returns:
        DataFrame with 13 BartTorvik columns + Team + Year.
    """
    date_str = SNAPSHOT_DATES.get(year)
    if date_str is None:
        logger.warning("No snapshot date for year %d", year)
        return pd.DataFrame()
    
    url = TIME_MACHINE_URL.format(date=date_str)
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", REQUEST_HEADERS["User-Agent"])
        req.add_header("Accept-Encoding", "identity")
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw_bytes = resp.read()
    except Exception as e:
        logger.error("Download failed for %s: %s", url, e)
        return pd.DataFrame()
    
    try:
        try:
            decompressed = gzip.decompress(raw_bytes)
        except (gzip.BadGzipFile, OSError):
            decompressed = raw_bytes
        records = json.loads(decompressed)
    except (json.JSONDecodeError, Exception) as e:
        logger.error("JSON parsing failed for %s: %s", url, e)
        return pd.DataFrame()
    
    if not isinstance(records, list) or len(records) == 0:
        logger.warning("Empty data for year %d", year)
        return pd.DataFrame()
    
    rows = []
    for rec in records:
        if not isinstance(rec, list) or len(rec) < 45:
            continue
        row = {col_name: rec[idx] for col_name, idx in COL_IDX.items()}
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df["Year"] = year
    
    # Standardize team names
    df["Team"] = df["Team"].astype(str).str.strip()
    
    # Ensure numeric columns
    numeric_cols = [c for c in df.columns if c not in ("Team", "Year")]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    logger.info("Downloaded %d teams for year %d", len(df), year)
    return df


def download_all_barttorvik(
    years: list[int] | None = None,
    output_path: str = "data_files/barttorvik_extended.csv",
) -> pd.DataFrame:
    """Download BartTorvik data for all years and save to CSV.
    
    Args:
        years: List of season years. Defaults to all available.
        output_path: Output CSV path.
        
    Returns:
        Combined DataFrame of all years.
    """
    years = years or sorted(SNAPSHOT_DATES.keys())
    frames = [download_barttorvik_snapshot(y) for y in years]
    frames = [f for f in frames if not f.empty]
    
    if not frames:
        logger.warning("No BartTorvik data downloaded.")
        return pd.DataFrame()
    
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s", len(df), output_path)
    return df
```

---

## 3. Matchup Interaction Features (11 columns)

These are the most novel features in sports-quant — they capture how team **styles interact** with each other, rather than just measuring team quality in isolation.

```python
# From sports-quant _data.py — compute_matchup_features()

MATCHUP_FEATURE_COLUMNS = [
    "offense_vs_defense_mismatch",
    "bart_offense_vs_defense_mismatch",
    "offense_defense_product",
    "bart_offense_defense_product",
    "tempo_mismatch_magnitude",
    "tempo_x_quality_interaction",
    "tempo_x_seed_interaction",
    "seed_upset_prior_centered",
    "seed_x_quality_gap",
    "quality_source_agreement",
    "sos_quality_interaction",
]


def compute_matchup_features(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Compute matchup interaction features from raw stats.
    
    These features capture how team styles interact — not just 
    which team is better, but HOW their strengths/weaknesses match up.
    
    Args:
        matchups_df: DataFrame with raw Team1/Team2 stats.
        
    Returns:
        DataFrame with 11 matchup interaction columns.
    """
    mf = pd.DataFrame(index=matchups_df.index)
    
    # --- Offense vs Defense Mismatches ---
    # How much does Team1's offense overpower Team2's defense (and vice versa)?
    # Positive = Team1 has offense/defense advantage
    
    # KenPom version
    t1_adjo = pd.to_numeric(matchups_df.get("AdjO", 0), errors="coerce").fillna(0)
    t1_adjd = pd.to_numeric(matchups_df.get("AdjD", 0), errors="coerce").fillna(0)
    t2_adjo = pd.to_numeric(matchups_df.get("AdjO_Team2", 0), errors="coerce").fillna(0)
    t2_adjd = pd.to_numeric(matchups_df.get("AdjD_Team2", 0), errors="coerce").fillna(0)
    
    mf["offense_vs_defense_mismatch"] = (t1_adjo - t2_adjd) - (t2_adjo - t1_adjd)
    
    # BartTorvik version
    bt1_oe = pd.to_numeric(matchups_df.get("Bart_AdjOE", 0), errors="coerce").fillna(0)
    bt1_de = pd.to_numeric(matchups_df.get("Bart_AdjDE", 0), errors="coerce").fillna(0)
    bt2_oe = pd.to_numeric(matchups_df.get("Bart_AdjOE_Team2", 0), errors="coerce").fillna(0)
    bt2_de = pd.to_numeric(matchups_df.get("Bart_AdjDE_Team2", 0), errors="coerce").fillna(0)
    
    mf["bart_offense_vs_defense_mismatch"] = (bt1_oe - bt2_de) - (bt2_oe - bt1_de)
    
    # --- Offense × Defense Products ---
    # Captures magnitude of matchup quality (high offense + weak defense = blowout risk)
    mf["offense_defense_product"] = (t1_adjo - t2_adjd) * (t2_adjo - t1_adjd)
    mf["bart_offense_defense_product"] = (bt1_oe - bt2_de) * (bt2_oe - bt1_de)
    
    # --- Tempo Mismatch ---
    # Large tempo mismatch = one team forced out of comfort zone
    t1_tempo = pd.to_numeric(matchups_df.get("AdjT", 0), errors="coerce").fillna(68)
    t2_tempo = pd.to_numeric(matchups_df.get("AdjT_Team2", 0), errors="coerce").fillna(68)
    
    mf["tempo_mismatch_magnitude"] = (t1_tempo - t2_tempo).abs()
    
    # --- Tempo × Quality Interaction ---
    # Does a tempo mismatch matter more when teams are close in quality?
    quality_gap = pd.to_numeric(matchups_df.get("AdjEM", 0), errors="coerce").fillna(0) - \
                  pd.to_numeric(matchups_df.get("AdjEM_Team2", 0), errors="coerce").fillna(0)
    mf["tempo_x_quality_interaction"] = mf["tempo_mismatch_magnitude"] * quality_gap.abs()
    
    # --- Tempo × Seed Interaction ---
    # Does tempo mismatch benefit the underdog?
    seed1 = pd.to_numeric(matchups_df.get("Seed1", 8), errors="coerce").fillna(8)
    seed2 = pd.to_numeric(matchups_df.get("Seed2", 8), errors="coerce").fillna(8)
    seed_diff = seed1 - seed2
    mf["tempo_x_seed_interaction"] = mf["tempo_mismatch_magnitude"] * seed_diff
    
    # --- Seed Upset Prior ---
    # Historical base rate for seed matchup outcomes, centered at 0
    SEED_MATCHUP_PRIORS = {
        (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
        (5, 12): 0.65, (6, 11): 0.62, (7, 10): 0.61, (8, 9): 0.52,
        (1, 8): 0.80, (1, 9): 0.85, (2, 7): 0.65, (2, 10): 0.70,
        (3, 6): 0.58, (3, 11): 0.65, (4, 5): 0.55, (4, 12): 0.65,
        (1, 4): 0.65, (1, 5): 0.70, (2, 3): 0.55, (2, 6): 0.62,
        (1, 2): 0.55, (1, 3): 0.60,
    }
    
    def get_seed_prior(s1, s2):
        """Look up historical win probability for seed matchup."""
        key = (min(s1, s2), max(s1, s2))
        prior = SEED_MATCHUP_PRIORS.get(key, 0.5)
        # If team1 is the higher seed (lower number), return prior
        # Otherwise return 1 - prior
        return prior if s1 <= s2 else 1 - prior
    
    mf["seed_upset_prior_centered"] = [
        get_seed_prior(s1, s2) - 0.5
        for s1, s2 in zip(seed1, seed2)
    ]
    
    # --- Seed × Quality Gap ---
    # Do quality ratings disagree with seeding? (potential upset signal)
    mf["seed_x_quality_gap"] = seed_diff * quality_gap
    
    # --- Quality Source Agreement ---
    # Do KenPom and BartTorvik agree on who's better?
    kenpom_gap = quality_gap
    bart_gap = bt1_oe - bt1_de - (bt2_oe - bt2_de)  # BartTorvik net efficiency gap
    mf["quality_source_agreement"] = kenpom_gap * bart_gap
    
    # --- SOS × Quality Interaction ---
    # Is a team's quality backed by a strong schedule?
    t1_sos = pd.to_numeric(matchups_df.get("SOS_NetRtg", 0), errors="coerce").fillna(0)
    t2_sos = pd.to_numeric(matchups_df.get("SOS_NetRtg_Team2", 0), errors="coerce").fillna(0)
    sos_diff = t1_sos - t2_sos
    mf["sos_quality_interaction"] = sos_diff * quality_gap
    
    return mf
```

---

## 4. Data Symmetrization

Training data symmetrization **doubles the training set** and eliminates positional bias. For each game (Team1 beats Team2), create a mirror row (Team2 loses to Team1).

With difference features, symmetrization is trivial: negate all features and flip the label.

```python
def symmetrize_training_data(
    X: pd.DataFrame, 
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Double training data by adding mirrored rows.
    
    For difference features: negate all values and flip labels.
    This eliminates any residual Team1/Team2 positional bias
    and doubles the effective training set size.
    
    Args:
        X: Difference feature DataFrame.
        y: Binary labels (1 = Team1 wins, 0 = Team2 wins).
        
    Returns:
        Tuple of (doubled X, doubled y).
    """
    # Mirror: negate features, flip label
    X_mirrored = X * -1
    y_mirrored = 1 - y
    
    X_combined = pd.concat([X, X_mirrored], ignore_index=True)
    y_combined = pd.concat([y, y_mirrored], ignore_index=True)
    
    return X_combined, y_combined
```

---

## 5. Feature Builder (FeatureLookup Pattern)

The sports-quant repo uses a `FeatureLookup` class that indexes by `(team, year)` and can build feature vectors for **arbitrary matchups** — not just games that actually occurred. This is critical for bracket simulation (predicting hypothetical future-round matchups).

```python
# feature_lookup.py — Immutable lookup for constructing matchup features

import pandas as pd
import numpy as np
from typing import Optional


class FeatureLookup:
    """Immutable lookup for constructing matchup feature vectors.
    
    Indexes team stats by (team_name, year) and can build feature vectors
    for any hypothetical matchup — not just games that actually occurred.
    Essential for bracket simulation forward prediction.
    """
    
    def __init__(
        self,
        kenpom_df: pd.DataFrame,
        barttorvik_df: Optional[pd.DataFrame] = None,
        adjustments: Optional[dict] = None,
    ):
        """Initialize lookup index.
        
        Args:
            kenpom_df: KenPom ratings DataFrame with Team and Year columns.
            barttorvik_df: BartTorvik ratings DataFrame (optional).
            adjustments: Dict of (team, year) -> adjusted stat values (e.g. injuries).
        """
        # Build KenPom index: (team_name, year) -> row dict
        self._kenpom_index = {}
        for _, row in kenpom_df.iterrows():
            key = (str(row.get("Team", "")).strip(), int(row.get("Year", 0)))
            self._kenpom_index[key] = row.to_dict()
        
        # Build BartTorvik index if provided
        self._bart_index = {}
        if barttorvik_df is not None:
            for _, row in barttorvik_df.iterrows():
                key = (str(row.get("Team", "")).strip(), int(row.get("Year", 0)))
                self._bart_index[key] = row.to_dict()
        
        # Apply adjustments (e.g. injury adjustments)
        self._adjustments = adjustments or {}
    
    def get_team(self, team: str, year: int) -> Optional[dict]:
        """Look up combined stats for a team/year."""
        key = (team, year)
        kenpom = self._kenpom_index.get(key)
        if kenpom is None:
            return None
        
        # Merge BartTorvik if available
        stats = dict(kenpom)
        bart = self._bart_index.get(key)
        if bart:
            stats.update(bart)
        
        # Apply adjustments
        adj = self._adjustments.get(key)
        if adj:
            stats.update(adj)
        
        return stats
    
    def build_difference_features(
        self, team1: str, team2: str, year: int,
        seed1: int = 8, seed2: int = 8,
    ) -> Optional[pd.DataFrame]:
        """Build difference feature vector for a matchup.
        
        Args:
            team1, team2: Team names.
            year: Season year.
            seed1, seed2: Tournament seeds.
            
        Returns:
            Single-row DataFrame with difference features, or None if lookup fails.
        """
        t1 = self.get_team(team1, year)
        t2 = self.get_team(team2, year)
        if t1 is None or t2 is None:
            return None
        
        # Compute KenPom diffs
        features = {}
        kenpom_pairs = [
            ("Rk", "diff_Rank"), ("NetRtg", "diff_AdjEM"),
            ("ORtg", "diff_AdjO"), ("DRtg", "diff_AdjD"),
            ("AdjT", "diff_AdjT"), ("SOS_NetRtg", "diff_SOS"),
        ]
        for col, out_name in kenpom_pairs:
            v1 = float(t1.get(col, 0) or 0)
            v2 = float(t2.get(col, 0) or 0)
            features[out_name] = v1 - v2
        
        # Compute BartTorvik diffs
        bart_pairs = [
            ("Bart_Rank", "diff_Bart_Rank"), ("Bart_AdjOE", "diff_Bart_AdjOE"),
            ("Bart_AdjDE", "diff_Bart_AdjDE"), ("Bart_Barthag", "diff_Barthag"),
            ("Bart_AdjT", "diff_Bart_AdjT"), ("Bart_SOS", "diff_Bart_SOS"),
            ("Bart_NCSOS", "diff_Bart_NCSOS"), ("Bart_EliteSOS", "diff_Bart_EliteSOS"),
            ("Bart_WAB", "diff_Bart_WAB"), ("Bart_QualO", "diff_Bart_QualO"),
            ("Bart_QualD", "diff_Bart_QualD"),
            ("Bart_QualBarthag", "diff_Bart_QualBarthag"),
        ]
        for col, out_name in bart_pairs:
            v1 = float(t1.get(col, 0) or 0)
            v2 = float(t2.get(col, 0) or 0)
            features[out_name] = v1 - v2
        
        features["diff_Seed"] = seed1 - seed2
        
        return pd.DataFrame([features])
```

---

## 6. Seed Matchup Priors

Historical win rates for every seed combination, used as Bayesian anchoring features:

```python
SEED_MATCHUP_PRIORS = {
    # Round of 64
    (1, 16): 0.99,  (2, 15): 0.94,  (3, 14): 0.85,  (4, 13): 0.79,
    (5, 12): 0.65,  (6, 11): 0.62,  (7, 10): 0.61,  (8, 9): 0.52,
    # Round of 32
    (1, 8): 0.80,   (1, 9): 0.85,   (2, 7): 0.65,   (2, 10): 0.70,
    (3, 6): 0.58,   (3, 11): 0.65,  (4, 5): 0.55,   (4, 12): 0.65,
    # Sweet 16
    (1, 4): 0.65,   (1, 5): 0.70,   (2, 3): 0.55,   (2, 6): 0.62,
    # Elite 8 / Final Four / Championship
    (1, 2): 0.55,   (1, 3): 0.60,
}
```

**Usage:** The `seed_upset_prior_centered` feature gives the model a Bayesian anchor: "historically, a 5-seed beats a 12-seed 65% of the time." The model can then learn to adjust from this baseline using team-specific quality metrics.

---

## 7. Combined Feature Pipeline (45 columns total)

The full pipeline assembles all three feature groups:

```python
def compute_combined_features(matchups_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full 45-column combined feature set.
    
    Combines:
    - 21 KenPom difference features
    - 13 BartTorvik difference features  
    - 11 Matchup interaction features
    
    Returns:
        DataFrame with 45 feature columns.
    """
    kenpom_diff = compute_kenpom_difference_features(matchups_df)     # 21 cols
    bart_diff = compute_barttorvik_difference_features(matchups_df)   # 13 cols  
    matchup = compute_matchup_features(matchups_df)                   # 11 cols
    
    return pd.concat([kenpom_diff, bart_diff, matchup], axis=1)
```

---

## 8. Team Name Standardization

A critical operational requirement — KenPom, BartTorvik, ESPN, and our training data all use different team name conventions:

```python
TEAM_NAME_MAPPING = {
    # Common variations that need standardization
    "N.C. State": "NC State",
    "North Carolina St.": "NC State",
    "UConn": "Connecticut",
    "UCONN": "Connecticut",
    "Saint Mary's": "Saint Mary's (CA)",
    "St. Mary's": "Saint Mary's (CA)", 
    "Miami FL": "Miami (FL)",
    "Miami (OH)": "Miami (OH)",
    "UCF": "Central Florida",
    "VCU": "Virginia Commonwealth",
    "LSU": "Louisiana State",
    "USC": "Southern California",
    "SMU": "Southern Methodist",
    "BYU": "Brigham Young",
    "UNLV": "Nevada-Las Vegas",
    "UAB": "Alabama-Birmingham",
    "Loyola Chicago": "Loyola (IL)",
    "Loyola-Chicago": "Loyola (IL)",
    # Add more as needed during integration
}

def standardize_team_name(name: str) -> str:
    """Standardize team name across data sources."""
    name = name.strip()
    return TEAM_NAME_MAPPING.get(name, name)
```

---

## Priority Implementation Order

1. **Difference features + symmetrization** — Highest ROI, simplest to implement
2. **Extended BartTorvik Time Machine download** — New data source, 13 columns
3. **Matchup interaction features** — Novel features that capture style mismatches
4. **FeatureLookup class** — Enables bracket simulation with arbitrary matchups
5. **Seed matchup priors** — Bayesian anchoring for seed-based predictions
6. **Team name standardization** — Required for cross-source data merging
