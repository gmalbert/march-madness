# Bracket Oracle — Architecture

## Overview
NCAA March Madness prediction and betting analytics platform. Predicts game outcomes using KenPom/BartTorvik adjusted efficiency ratings, Monte Carlo bracket simulation, and betting market value detection.

## Data Flow
```
KenPom (adj. efficiency)    BartTorvik    ESPN API    The Odds API
        ↓                       ↓             ↓             ↓
analyze_kenpom.py        download_barttorvik.py  data_collection.py
        ↓                       ↓
        ↓           data_files/ (prediction CSVs, efficiency data)
        ↓                       ↓
        └───────────→ betting_models.py (model training + prediction)
                                ↓
                        bracket_simulation.py (Monte Carlo)
                                ↓
                        predictions.py (Streamlit entry)
                                ↓
                    scripts/export_best_bets.py
                                ↓
                    data_files/best_bets_today.json
```

## ML Models
- **Primary**: Logistic regression + XGBoost on adjusted efficiency differentials
- **Features**: AdjOE, AdjDE, tempo, seed, H2H, recent form, tournament experience
- **Bracket Simulation** (`bracket_simulation.py`): Monte Carlo simulation of full 68-team bracket
- **Efficiency Ratings**: KenPom (primary), BartTorvik (secondary — additional tempo data)

## Tournament Structure
- 68-team field: First Four → Round of 64 → Elite Eight → Final Four → Championship
- All tournament games played on neutral courts (no home-court advantage)
- Seed matchup probabilities: historical upset rates per seed pairing

## API Integrations
| Source | Purpose | Key |
|--------|---------|-----|
| KenPom | Adjusted efficiency ratings | Requires subscription |
| BartTorvik | Ratings + tempo data | None (scraped) |
| ESPN API | Scores, schedules, team stats | None (public) |
| The Odds API | `basketball_ncaab` markets | `ODDS_API_KEY` |

## Key Components
- `predictions.py` — entry, `st.set_page_config`
- `data_collection.py` — `fetch_games()`, `fetch_betting_lines()`, `fetch_adjusted_efficiency()`
- `betting_models.py` — model training and prediction logic
- `bracket_simulation.py` — Monte Carlo bracket simulation
- `analyze_kenpom.py` — KenPom data processing
- `download_barttorvik.py` — BartTorvik data fetcher
- `footer.py` — `add_betting_oracle_footer()`
- `scripts/export_best_bets.py` — writes `data_files/best_bets_today.json`

## Storage
- `data_files/` — prediction CSVs, model artifacts, odds snapshots
- `data_files/best_bets_today.json` — Sports Picks Grid feed
- `data_files/logo.png` — app logo

## Bet Types
- `moneyline` — outright game winner
- `spread` — point spread
- `total` — over/under total points
