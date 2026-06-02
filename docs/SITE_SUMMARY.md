> **AI Onboarding Guide** — See also the project docs folder for detailed architecture documentation.

# March Madness (College Basketball) — Site Summary

## What This App Does

Streamlit app predicting NCAA March Madness tournament outcomes using ML models trained on 10 years of historical data (2016–2025). Combines KenPom efficiency ratings and BartTorvik adjusted efficiency with CBBD betting line data to identify value bets, detect upset candidates, and provide spread/moneyline predictions with Kelly Criterion sizing.

## Quick Start

```bash
# 1. Activate virtual environment
.\.venv\Scripts\Activate.ps1        # Windows
source .venv/bin/activate           # macOS/Linux

# 2. Run the app
streamlit run predictions.py
```

## Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit (multi-page) |
| ML | XGBoost + RF + GB + LR (soft-voting ensemble, 4 models) |
| Efficiency ratings | KenPom, BartTorvik (scraped / imported) |
| Data source | College Basketball Data API (CBBD) |
| Data storage | CSV + Parquet (`data_files/cache/`) |
| Visualization | Plotly |

## Key Files

| File | Purpose |
|---|---|
| `predictions.py` | Main Streamlit dashboard — tournament bracket, betting analysis, upset detection |
| `data_collection.py` | CBBD API integration — fetches games, odds, team stats |
| `model_training.py` | Ensemble training (XGBoost, RF, GB, LR) for spread/moneyline/totals |
| `feature_engineering.py` | Builds 11+ features from CBBD + KenPom + BartTorvik |
| `underdog_value.py` | Identifies profitable underdog opportunities via edge calculation |
| `upset_prediction.py` | Cinderella candidate detection |

## Data Flow

1. **Raw data**: `data_collection.py` → CBBD API → cached CSV in `data_files/cache/`
2. **Efficiency ratings**: KenPom / BartTorvik data (manually updated or scraped) merged into features
3. **Feature engineering**: `feature_engineering.py` → 11+ features (NetRtg, ORtg, DRtg, AdjT, Luck, SOS, Pythagorean expectations)
4. **Training**: `model_training.py` → soft-voting ensemble → saved to `data_files/cache/`
5. **Predictions**: model scores upcoming matchups → edge = model probability vs Vegas spread/line
6. **UI**: Streamlit renders tournament bracket, upset watch lists, value bet table

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `CBBD_API_KEY` | College Basketball Data API | Required |
| `ODDS_API_KEY` | The Odds API — live Vegas spreads/totals | Optional |
| KenPom credentials | If scraping KenPom directly (paid subscription) | Optional |

## Key Features in the UI

- Tournament bracket with seed-based matchup history
- Upset prediction (Cinderella detection) per round
- Value bet table with edge calculations and Kelly sizing
- Live line movement tracker (documented in `docs/LIVE_LINE_MOVEMENT_TRACKER_README.md`, partially implemented)

## Critical Conventions

- Data freshness: CBBD data updates during the season; KenPom/BartTorvik require separate refresh
- Rolling stats must use `shift(1)` to prevent leakage when computing team form features
- Ensemble uses soft voting — each model outputs probabilities; final prediction is weighted average

## Common Gotchas

- KenPom access requires a paid subscription; without it, fall back to BartTorvik (free)
- The interactive bracket visualizer is documented in `docs/BRACKET_SIMULATOR.md` but not yet fully implemented in the UI
- Live line movement tracker captures only opening lines currently — real-time polling is not implemented
- Kelly sizing is calculated but not persisted across tournament sessions
