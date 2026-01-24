# KenPom Fields and Suggested Derived Features

This document describes the columns extracted from KenPom (`data_files/kenpom_ratings.csv`) and suggests derived features useful for modeling March Madness games.

## Column definitions
- `Rk` — KenPom overall rank (1 = best).
- `Team` — School name (string).
- `Conf` — Conference abbreviation (string).
- `W-L` — Win–loss record (string, e.g., `18-1`). Convert to wins, losses, or win percentage as needed.
- `NetRtg` — Adjusted efficiency margin (points per 100 possessions): Adj ORtg − Adj DRtg. Numeric (may include `+` sign).
- `ORtg` — Adjusted offensive rating (points per 100 possessions). Numeric.
- `ORtg_Rank` — Ordinal rank for `ORtg` (1 = best).
- `DRtg` — Adjusted defensive rating (points allowed per 100 possessions). Lower is better.
- `DRtg_Rank` — Ordinal rank for `DRtg`.
- `AdjT` — Adjusted tempo (estimated possessions per game). Numeric.
- `AdjT_Rank` — Ordinal rank for tempo.
- `Luck` — KenPom luck metric (how much observed record deviates from expected performance). Positive = luckier.
- `Luck_Rank` — Ordinal rank for luck.
- `SOS_NetRtg` — Strength-of-schedule measured as opponents' NetRtg.
- `SOS_NetRtg_Rank` — Rank for SOS NetRtg.
- `SOS_ORtg` — SOS measured by opponents' ORtg.
- `SOS_ORtg_Rank` — Rank for SOS ORtg.
- `SOS_DRtg` — SOS measured by opponents' DRtg.
- `SOS_DRtg_Rank` — Rank for SOS DRtg.
- `NCSOS_NetRtg` — Non-conference strength-of-schedule NetRtg.
- `NCSOS_NetRtg_Rank` — Rank for non-conference SOS NetRtg.

Notes:
- Ranks are ordinal (1 is best). Convert to numeric `int` when using in models.
- Many fields include formatting (leading `+`, percent-like text). Clean to floats/ints before modeling.

## Recommended data types / cleaning
- Strip `+` signs from `NetRtg` and other signed numbers and convert to `float`.
- Convert `W-L` into `wins`, `losses`, and `win_pct` (`wins / (wins+losses)`).
- Cast all `_Rank` columns to `int`.
- Impute missing numeric fields with sensible defaults (median by conference or overall) or drop rows if appropriate.

## Suggested derived features
Below are features that frequently improve predictive models for individual games.

- Value differences (team A minus team B):
  - `NetRtg_diff = NetRtg_A - NetRtg_B`
  - `ORtg_diff`, `DRtg_diff`, `AdjT_diff`, `Luck_diff`, `SOS_NetRtg_diff`, etc.

- Rank differences:
  - `NetRtg_rank_diff` (if you compute a rank), `ORtg_rank_diff`, etc. Rank gaps capture ordinal separation.

- Relative / normalized features:
  - Z-score each metric across Division I teams, then use z-score differences.

- Tempo-adjusted interactions:
  - `NetRtg * AdjT` or `ORtg_diff * AdjT_diff` to capture matchup effects when styles clash.

- Strength-of-schedule adjustments:
  - `NetRtg_adj_for_SOS = NetRtg - SOS_NetRtg` to estimate performance relative to opponent quality.

- Recent performance signals (if you have game-by-game data):
  - `NetRtg_last5`, `NetRtg_trend` (slope), `Luck_last5`.

- Upset/variance features:
  - `Luck_diff`, `Rank_gap`, volatility measures (std dev of recent NetRtg) to model upset likelihood.

- Simple probability transforms:
  - Convert rating differences into probabilities using a logistic transform: p = 1 / (1 + exp(-k * diff)). Calibrate `k` on historical outcomes.

- Home/neutral adjustments:
  - Apply a home-court shift (e.g., add ~3 points to NetRtg for home team) or model `home` as an indicator.

## Example quick-start transformations (pandas)
```python
import pandas as pd

df = pd.read_csv('data_files/kenpom_ratings.csv')

# clean numeric
df['NetRtg'] = df['NetRtg'].str.replace('+', '', regex=False).astype(float)
df['ORtg'] = df['ORtg'].astype(float)
for c in df.columns:
    if c.endswith('_Rank'):
        df[c] = pd.to_numeric(df[c], errors='coerce').astype('Int64')

# expand W-L
wl = df['W-L'].str.split('-', expand=True).astype(int)
df['wins'], df['losses'] = wl[0], wl[1]
df['win_pct'] = df['wins'] / (df['wins'] + df['losses'])
```

## Modeling tips
- Prefer differences (A − B) of cleaned numeric features as primary inputs to binary win/loss classifiers.
- Regularize interactions (e.g., with elastic net) to avoid overfitting when including many cross features.
- Calibrate predicted probabilities (Platt scaling or isotonic regression) for better simulation/summing of bracket probabilities.

If you want, I can:
- Add a small loader script `data_tools/kenpom_loader.py` that reads, cleans, and outputs a canonical CSV for modeling, or
- Create `docs/kenpom_fields.md` with expanded examples for feature engineering (game-level joins, merging with BartTorvik, name canonicalization).

---
Generated automatically as a concise reference for the project.
