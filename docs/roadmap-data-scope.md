# Roadmap: Data Scope (Basketball Betting)

*What basketball data to gather for March Madness betting predictions.*

## Recommended Historical Range
- **Minimum**: 5 years (2021-2025)
- **Recommended**: 10 years (2016-2025)
- **Note**: Data availability may vary; tournament structure consistent since 2011

## Core Data Sets for Betting Models

### 1. Tournament Games with Results
Essential for training win prediction models.

```python
def fetch_all_tournament_games(start_year=2016, end_year=2025):
    """Fetch all March Madness tournament games."""
    all_games = []
    for year in range(start_year, end_year + 1):
        games = fetch_games(year, season_type="postseason")
        all_games.extend(games)
    return all_games
```

### 2. Betting Lines (Spreads, O/U, Moneyline)
Critical for spread and total predictions.

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

### 3. Team Season Statistics
For building team strength profiles.

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

### 4. Adjusted Efficiency Ratings
Best predictors for tournament success.

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

### 5. Rankings
For seeding and public perception factors.

```python
def fetch_rankings(year: int, week: int = None):
    """Fetch poll rankings."""
    with get_api_client() as api_client:
        rankings_api = cbbd.RankingsApi(api_client)
        return rankings_api.get_rankings(year=year, week=week)
```

## Data Priority for Betting

| Priority | Data Set | Bet Types | Years |
|----------|----------|-----------|-------|
| **P0** | Betting Lines | All | 10 |
| **P0** | Game Results | All | 10 |
| **P0** | Adjusted Efficiency | All | 10 |
| **P1** | Team Season Stats | Spread, O/U | 10 |
| **P1** | Four Factors | All | 10 |
| **P2** | Rankings | Moneyline | 10 |
| **P2** | Player Stats | Props | 5 |

## Target Prediction Types

1. **Winner Prediction (Moneyline)**
   - Binary classification: which team wins
   - Key data: efficiency ratings, rankings

2. **Spread Prediction (ATS)**
   - Regression: predict margin of victory
   - Compare to betting spread
   - Key data: efficiency differential, historical ATS

3. **Over/Under Prediction**
   - Regression: predict total points
   - Compare to betting total
   - Key data: tempo, offensive/defensive efficiency

4. **Underdog Value Bets**
   - Identify underdogs with >expected probability
   - Key data: efficiency vs seed, recent form

## Storage Structure

```
data_files/
├── raw/
│   ├── games_2024.json
│   ├── lines_2024.json
│   └── efficiency_2024.json
├── processed/
│   ├── training_data.csv
│   └── betting_features.csv
└── cache/
```

## Next Steps
- See `roadmap-betting-features.md` for feature engineering
- See `roadmap-betting-models.md` for model approaches
