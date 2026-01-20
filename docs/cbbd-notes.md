# CBBD Python Library Notes

*These notes are prepared for Copilot instructions to assist the agent in reviewing the CBBD library for March Madness predictions.*

## Overview
The CBBD (College Basketball Data) Python library is a wrapper for the College Basketball Data API. It provides access to college basketball datasets and analytics, including games, teams, players, stats, rankings, and **betting lines**. This is the correct library for March Madness tournament predictions.

- **API Version**: 1.22.4
- **Package Version**: 1.22.4
- **Host**: https://api.collegebasketballdata.com
- **Source**: https://github.com/CFBD/cbbd-python

## Requirements
- Python 3.7 or higher

## Installation
```bash
pip install cbbd
```

## Authentication
API key from [CollegeBasketballData.com](https://collegebasketballdata.com/). Uses Bearer token authorization.

```python
import os
import cbbd

configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com",
    access_token=os.environ["CBBD_API_KEY"]
)
```

## Key API Endpoints

### Games
- `get_games(season=2024, season_type="regular")`: Game data with scores
- `get_game_teams`: Team box scores
- `get_game_players`: Player box scores
- `get_broadcasts`: Media/broadcast info

### Betting Lines (Critical for Project)
- `get_lines()`: Betting lines (spreads, over/unders, moneylines) - returns all available data
- `get_providers`: Line providers (Vegas, Bovada, etc.)

### Teams
- `get_teams`: Team information
- `get_team_roster`: Team rosters

### Stats
- `get_team_season_stats(season=2024)`: Team season statistics
- `get_team_season_shooting_stats`: Shooting breakdowns
- `get_player_season_stats`: Player season stats
- `get_player_season_shooting_stats`: Player shooting stats

### Ratings
- `get_adjusted_efficiency(year=2024)`: Adjusted offensive/defensive efficiency (KenPom-style)
- `get_srs`: Simple Rating System

### Rankings
- `get_rankings`: Poll rankings (AP, Coaches, etc.)

### Plays
- `get_plays`: Play-by-play data
- `get_plays_by_tournament`: Tournament-specific plays

### Other
- `get_conferences`: Conference info
- `get_recruits`: Recruiting data
- `get_draft_picks`: NBA draft picks
- `get_lineup_stats_by_game`: Lineup analysis

## Key Data Models

### GameLines (Dictionary Structure)
Betting lines are returned as dictionaries with these keys:
- `gameId`: Unique game identifier
- `season`: Year of the season
- `seasonType`: "regular" or "postseason"
- `startDate`: Game date/time
- `homeTeamId`, `homeTeam`: Home team info
- `awayTeamId`, `awayTeam`: Away team info
- `homeScore`, `awayScore`: Final scores (if completed)
- `lines`: Array of betting lines, each containing:
  - `spread`: Point spread
  - `overUnder`: Total points line
  - `homeMoneyline`, `awayMoneyline`: Moneyline odds
  - `provider`: Source of the line

### AdjustedEfficiencyInfo (Dictionary Structure)
Efficiency ratings returned as dictionaries with these keys:
- `season`: Year of the season
- `teamId`: Unique team identifier
- `team`: Team name
- `conference`: Conference name
- `offensiveRating`: Offensive efficiency rating
- `defensiveRating`: Defensive efficiency rating
- `netRating`: Net efficiency rating (offensive - defensive)
- `rankings`: Dictionary with rankings for net, offense, and defense

### TeamSeasonStats
Comprehensive team stats:
- Points, rebounds, assists
- Field goal percentages
- Four factors (eFG%, TO%, ORB%, FTR)

## Usage Example

```python
import cbbd
from cbbd.rest import ApiException

configuration = cbbd.Configuration(
    access_token=os.environ["CBBD_API_KEY"]
)

with cbbd.ApiClient(configuration) as api_client:
    # Get betting lines (returns all available data)
    lines_api = cbbd.LinesApi(api_client)
    lines = lines_api.get_lines()
    
    # Get games for a specific season
    games_api = cbbd.GamesApi(api_client)
    games = games_api.get_games(season=2024, season_type="regular")
    
    # Get team stats for a season
    stats_api = cbbd.StatsApi(api_client)
    team_stats = stats_api.get_team_season_stats(season=2024)
    
    # Get efficiency ratings
    ratings_api = cbbd.RatingsApi(api_client)
    efficiency = ratings_api.get_adjusted_efficiency(year=2024)
    
    for game in lines[:5]:  # Show first 5 games
        print(f"{game['homeTeam']} vs {game['awayTeam']}")
        if game.get('lines'):
            line = game['lines'][0]
            print(f"  Spread: {line.get('spread', 'N/A')}, O/U: {line.get('overUnder', 'N/A')}")
```

## Notes for March Madness Betting Predictions
- Use `season_type="postseason"` for tournament games
- Betting lines include spread, over/under, and moneyline
- Adjusted efficiency ratings are key predictors
- Four factors correlate strongly with tournament success

## API Parameter Corrections (Based on Testing)
- `get_games()`: Use `season` parameter (not `year`)
- `get_team_season_stats()`: Use `season` parameter (not `year`)
- `get_lines()`: No parameters needed - returns all available betting data
- `get_adjusted_efficiency()`: Uses `year` parameter correctly
- All data is returned as dictionaries, not objects with attributes
- Access data using dictionary keys like `game['homeTeam']` instead of `game.home_team`
