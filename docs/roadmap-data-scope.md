# Roadmap: Data Scope

*What data to gather and how far back.*

## Recommended Historical Range
- **Minimum**: 5 years (2021-2025)
- **Recommended**: 10 years (2016-2025)
- **Maximum useful**: 15 years (2011-2025)

Older data may have less relevance due to rule changes, conference realignments, and evolving play styles.

## Core Data Sets

### 1. Games Data
Essential for win/loss records and score differentials.

```python
def fetch_all_games(start_year=2016, end_year=2025):
    """Fetch comprehensive games data."""
    with get_api_client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        all_games = []
        for year in range(start_year, end_year + 1):
            games = games_api.get_games(year=year)
            all_games.extend(games)
        return all_games
```

### 2. Team Statistics
Season-level team performance metrics.

```python
def fetch_team_stats(year: int):
    """Fetch team season stats."""
    with get_api_client() as api_client:
        stats_api = cfbd.StatsApi(api_client)
        return stats_api.get_team_stats(year=year)
```

### 3. Advanced Stats
EPA, PPA, success rates, and efficiency metrics.

```python
def fetch_advanced_stats(year: int):
    """Fetch advanced season statistics."""
    with get_api_client() as api_client:
        stats_api = cfbd.StatsApi(api_client)
        return stats_api.get_advanced_season_stats(year=year)
```

### 4. Rankings and Ratings
Poll rankings, Elo, SP+, SRS ratings.

```python
def fetch_ratings(year: int):
    """Fetch multiple rating systems."""
    with get_api_client() as api_client:
        ratings_api = cfbd.RatingsApi(api_client)
        return {
            "elo": ratings_api.get_elo(year=year),
            "sp": ratings_api.get_sp(year=year),
            "srs": ratings_api.get_srs(year=year)
        }
```

### 5. Recruiting Data
Team talent and recruiting rankings.

```python
def fetch_recruiting(year: int):
    """Fetch recruiting rankings."""
    with get_api_client() as api_client:
        recruiting_api = cfbd.RecruitingApi(api_client)
        teams_api = cfbd.TeamsApi(api_client)
        return {
            "recruiting": recruiting_api.get_team_recruiting_rankings(year=year),
            "talent": teams_api.get_talent(year=year)
        }
```

## Secondary Data Sets

### 6. Betting Lines
Historical spreads and over/unders.

```python
def fetch_betting_lines(year: int):
    """Fetch betting data."""
    with get_api_client() as api_client:
        betting_api = cfbd.BettingApi(api_client)
        return betting_api.get_lines(year=year)
```

### 7. Player Stats
For tracking key player performance.

```python
def fetch_player_stats(year: int):
    """Fetch player season stats."""
    with get_api_client() as api_client:
        stats_api = cfbd.StatsApi(api_client)
        return stats_api.get_player_season_stats(year=year)
```

## Data Collection Priorities

| Priority | Data Set | Years | Use Case |
|----------|----------|-------|----------|
| High | Games | 10 | Win/loss, scores |
| High | Advanced Stats | 10 | Efficiency metrics |
| High | Ratings (SP+, Elo) | 10 | Team strength |
| Medium | Team Stats | 10 | Traditional stats |
| Medium | Recruiting | 10 | Talent evaluation |
| Low | Betting Lines | 5 | Market expectations |
| Low | Player Stats | 5 | Key player impact |

## Storage Recommendations
- Store raw JSON in `data_files/raw/`
- Store processed CSV/Parquet in `data_files/processed/`
- Use year-based naming: `games_2024.json`, `stats_2024.csv`

## Next Steps
- See `roadmap-features.md` for feature engineering
- See `roadmap-modeling.md` for model building
