# Roadmap: Data Scope (Basketball Betting)

*What basketball data to gather for March Madness betting predictions.*

## Recommended Historical Range
- **Minimum**: 5 years (2021-2025)
- **Recommended**: 10 years (2016-2025)
- **Note**: Data availability may vary; tournament structure consistent since 2011

## Core Data Sets for Betting Models

### 1. ✅ Tournament Games with Results
Essential for training win prediction models. **COMPLETED**

```python
def fetch_all_tournament_games(start_year=2016, end_year=2025):
    """Fetch all March Madness tournament games."""
    all_games = []
    for year in range(start_year, end_year + 1):
        games = fetch_games(year, season_type="postseason")
        all_games.extend(games)
    return all_games
```

*Status: ✅ Implemented - 4,623 weighted games (2016-2025), tournament games weighted 5x*

### 2. ❌ Betting Lines (Spreads, O/U, Moneyline)
Critical for spread and total predictions. **NOT IMPLEMENTED**

```python
def fetch_historical_lines(start_year=2016, end_year=2025):
    """Fetch historical betting lines for tournament games."""
    all_lines = []
    for year in range(start_year, end_year + 1):
        lines = fetch_betting_lines(year, "postseason")
        all_lines.extend(lines)
    return all_lines
```

**Key fields from betting lines:**
- `spread`: Point spread (favorite gives points)
- `over_under`: Total points line
- `home_moneyline`, `away_moneyline`: Moneyline odds
- `provider`: Source (consensus, individual books)

*Status: ❌ No betting lines data available in CBBD API*

### 3. ✅ Team Season Statistics
For building team strength profiles. **COMPLETED**

```python
def fetch_team_season_data(year: int):
    """Fetch comprehensive team stats."""
    with get_api_client() as api_client:
        stats_api = cbbd.StatsApi(api_client)
        return stats_api.get_team_season_stats(year=year)
```

**Key metrics:**
- Points per game (offense/defense)
- Field goal percentages (FG%, 3P%, FT%)
- Rebounds, assists, turnovers
- Four Factors: eFG%, TO%, ORB%, FTRate

*Status: ✅ Implemented - 700 teams with comprehensive stats (2025 season)*

### 4. ✅ Adjusted Efficiency Ratings
Best predictors for tournament success. **COMPLETED**

```python
def fetch_efficiency_ratings(year: int):
    """Fetch adjusted efficiency ratings."""
    with get_api_client() as api_client:
        ratings_api = cbbd.RatingsApi(api_client)
        return ratings_api.get_adjusted_efficiency(year=year)
```

**Key metrics:**
- Adjusted Offensive Efficiency (points per 100 possessions)
- Adjusted Defensive Efficiency
- Net Efficiency Rating
- Tempo (pace of play)

*Status: ✅ Implemented - 4,593 efficiency ratings (2025 season)*

### 5. ❌ Rankings
For seeding and public perception factors. **NOT IMPLEMENTED**

```python
def fetch_rankings(year: int, week: int = None):
    """Fetch poll rankings."""
    with get_api_client() as api_client:
        rankings_api = cbbd.RankingsApi(api_client)
        return rankings_api.get_rankings(year=year, week=week)
```

*Status: ❌ Not implemented - could enhance moneyline predictions*

## Data Priority for Betting

| Priority | Data Set | Bet Types | Years | Status |
|----------|----------|-----------|-------|--------|
| **P0** | ❌ Betting Lines | All | 10 | Not available in CBBD API |
| **P0** | ✅ Game Results | All | 10 | 4,623 tournament games (2016-2025) |
| **P0** | ✅ Adjusted Efficiency | All | 10 | 4,593 ratings across seasons |
| **P1** | ✅ Team Season Stats | Spread, O/U | 10 | 700 teams with comprehensive stats |
| **P1** | ✅ Four Factors | All | 10 | eFG%, TO%, ORB%, FTR extracted |
| **P2** | ❌ Rankings | Moneyline | 10 | Not implemented |
| **P2** | ❌ Player Stats | Props | 5 | Not implemented |

## Target Prediction Types

1. **✅ Winner Prediction (Moneyline)**
   - Binary classification: which team wins
   - Key data: efficiency ratings, rankings
   - *Status: ✅ Implemented - 71.0% accuracy (68.6%-74.8% range)*

2. **✅ Spread Prediction (ATS)**
   - Regression: predict margin of victory
   - Compare to betting spread
   - Key data: efficiency differential, historical ATS
   - *Status: ✅ Implemented - 12.74 MAE (12.24-13.22 range)*

3. **✅ Over/Under Prediction**
   - Regression: predict total points
   - Compare to betting total
   - Key data: tempo, offensive/defensive efficiency
   - *Status: ✅ Implemented - 16.58 MAE (15.03-17.38 range)*

4. **❌ Underdog Value Bets**
   - Identify underdogs with >expected probability
   - Key data: efficiency vs seed, recent form
   - *Status: ❌ Not specifically implemented*

## Storage Structure

```
data_files/
├── models/           ✅
│   ├── *_xgboost.joblib
│   ├── *_random_forest.joblib
│   ├── *_linear_regression.joblib
│   ├── *_metrics.json
│   └── *_scaler.joblib
├── cache/            ✅
│   ├── efficiency_2025.json
│   ├── team_stats_2025.json
│   └── historical_data.json.gz
├── espn_cbb_current_season.csv    ✅
└── training_data_weighted.csv     ✅
```

*Status: ✅ Implemented - Complete data pipeline with caching, models, and training data*

## Summary of Completed Work

**✅ Core Infrastructure:**
- Historical tournament data collection (2016-2025)
- Team efficiency ratings and statistics
- Weighted training dataset (regular + tournament games)
- ML model training pipeline (XGBoost, Random Forest, Linear/Logistic Regression)
- Real-time predictions with Streamlit UI
- Team name normalization for ESPN ↔ CBBD compatibility

**✅ Prediction Models:**
- Moneyline: 71.0% accuracy
- Spread: 12.74 MAE (points)
- Total: 16.58 MAE (points)

**❌ Missing Components:**
- Historical betting lines (not available in CBBD API)
- Team rankings data
- Player-level statistics
- Underdog value bet identification

**🎯 System Status:** Production-ready for tournament predictions using efficiency-based modeling
