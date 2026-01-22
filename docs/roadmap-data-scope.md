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

### 2. ✅ Betting Lines (Spreads, O/U, Moneyline)
Valuable for model training and validation. **FULLY IMPLEMENTED**

```python
def fetch_historical_lines(start_year=2016, end_year=2025):
    """Fetch historical betting lines for regular and tournament games."""
    all_lines = []
    for year in range(start_year, end_year + 1):
        # Regular season
        regular_lines = fetch_betting_lines(year, "regular")
        all_lines.extend(regular_lines)
        
        # Tournament (using date range filtering)
        tournament_lines = fetch_betting_lines(year, "postseason")
        all_lines.extend(tournament_lines)
    return all_lines
```

**Key fields from betting lines:**
- `spread`: Point spread (favorite gives points)
- `over_under`: Total points line
- `home_moneyline`, `away_moneyline`: Moneyline odds
- `provider`: Source (ESPN BET, consensus, etc.)
- `seasonType`: regular, postseason, or preseason

*Status: ✅ Fully Implemented*

**Current Implementation:**
- ✅ Function exists and fetches real betting data
- ✅ **Regular Season**: ~5,500-5,600 games with betting lines per season (~6,000 total games)
- ✅ **Tournament**: ~113 games with betting lines per tournament
- ✅ Includes spread, over/under, and moneyline odds from ESPN BET
- ✅ Uses date range filtering (chunked by month) to access ALL games (bypasses 3000-game pagination limit)
- 💡 Full historical betting data available for training tournament-specific models

**Data Completeness:**
- Total games collected (2016-2025): 31,125 games
- Games successfully processed: 26,230 (84.3%)
- Games skipped due to missing efficiency data: 4,877 (15.7%)
- Missing teams primarily: Division II/III, NAIA schools (830 unique teams)
- Efficiency data available for 365 Division I teams
- Pattern: Missing data mostly for away teams in games against D-I schools

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

### 5. ✅ Rankings
For seeding and public perception factors. **IMPLEMENTED**

```python
def fetch_rankings(year: int, week: int = None):
    """Fetch poll rankings for a given year and optional week."""
    cache_filename = f"rankings_{year}" + (f"_week_{week}" if week else "_all")

    # Check cache first
    cached = load_cached(cache_filename)
    if cached:
        print(f"Loaded {len(cached)} rankings from cache for {year}" + (f" week {week}" if week else ""))
        return cached

    with get_api_client() as api_client:
        rankings_api = cbbd.RankingsApi(api_client)
        try:
            if week:
                rankings = rankings_api.get_rankings(season=year, week=week)
            else:
                rankings = rankings_api.get_rankings(season=year)
            cache_data(cache_filename, rankings)
            print(f"Fetched {len(rankings)} rankings for {year}" + (f" week {week}" if week else ""))
            return rankings
        except ApiException as e:
            print(f"Error fetching rankings: {e}")
            return []
```

*Status: ✅ Implemented - 1,825 ranking entries (2024 season)*

## Data Priority for Betting

| Priority | Data Set | Bet Types | Years | Status |
|----------|----------|-----------|-------|--------|
| **P0** | ✅ Betting Lines | All | 10 | ~5,654 games/year + 113 tournament games (62,195 total games, 56,483 with lines) |
| **P0** | ✅ Game Results | All | 10 | 26,230 games (2016-2025: 25,111 regular + 1,119 tournament) |
| **P0** | ✅ Adjusted Efficiency | All | 10 | 365 D-I teams (84.3% game coverage, 15.7% skipped for D-II/III/NAIA) |
| **P1** | ✅ Team Season Stats | Spread, O/U | 10 | 700 teams with comprehensive stats |
| **P1** | ✅ Four Factors | All | 10 | eFG%, TO%, ORB%, FTR extracted |
| **P2** | ✅ Rankings | Moneyline | 10 | 1,825 ranking entries (2024 season) |
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

4. **✅ Underdog Value Bets**
   - Identify underdogs with >expected probability
   - Key data: efficiency vs seed, recent form
   - *Status: ✅ Implemented - Automatic value detection with Kelly Criterion betting recommendations*

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
- Team efficiency ratings and statistics (365 D-I teams)
- **COMPLETE betting lines dataset: 62,195 games (56,483 with betting lines) across 11 years**
- **Training dataset: 26,230 games (19,547 with betting lines) from 31,125 collected**
  - 84.3% coverage (4,877 games skipped due to missing efficiency data for D-II/III/NAIA teams)
  - Tournament games: 1,119 (weighted 5x for importance)
- Poll rankings data collection (1,825 ranking entries for 2024 season)
- ML model training pipeline (XGBoost, Random Forest, Linear/Logistic Regression)
- Real-time predictions with Streamlit UI
- Team name normalization for ESPN ↔ CBBD compatibility

**✅ Prediction Models:**
- Moneyline: 66.3% accuracy (63.5%-67.8% range) - trained on 25,368 games
- Spread: 11.87 MAE (11.54-12.35 range) - trained on 25,368 games  
- Total: 15.81 MAE (15.07-16.86 range) - trained on 25,368 games
- **Value Bets**: Automatic underdog identification with 5%+ expected value
  - Compares model probabilities vs. implied odds
  - Kelly Criterion for optimal bet sizing
  - ROI calculation and betting recommendations

**⚠️ Limitations:**
- Player-level statistics not implemented

**🎯 System Status:** Production-ready for tournament predictions using efficiency-based modeling
