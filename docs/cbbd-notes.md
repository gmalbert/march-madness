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
- `get_games`: Game data with scores
- `get_game_teams`: Team box scores
- `get_game_players`: Player box scores
- `get_broadcasts`: Media/broadcast info

### Betting Lines (Critical for Project)
- `get_lines`: Betting lines (spreads, over/unders, moneylines)
- `get_providers`: Line providers (Vegas, Bovada, etc.)

### Teams
- `get_teams`: Team information
- `get_team_roster`: Team rosters

### Stats
- `get_team_season_stats`: Team season statistics
- `get_team_season_shooting_stats`: Shooting breakdowns
- `get_player_season_stats`: Player season stats
- `get_player_season_shooting_stats`: Player shooting stats

### Ratings
- `get_adjusted_efficiency`: Adjusted offensive/defensive efficiency (KenPom-style)
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

### GameLines
Contains betting information:
- Spread
- Over/Under (total)
- Moneyline odds
- Provider info

### AdjustedEfficiencyInfo
Efficiency ratings for modeling:
- Adjusted offensive efficiency
- Adjusted defensive efficiency
- Rankings

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
    # Get betting lines
    lines_api = cbbd.LinesApi(api_client)
    lines = lines_api.get_lines(year=2025, season_type="postseason")
    
    for game in lines:
        print(f"{game.home_team} vs {game.away_team}")
        for line in game.lines:
            print(f"  Spread: {line.spread}, O/U: {line.over_under}")
```

## Notes for March Madness Betting Predictions
- Use `season_type="postseason"` for tournament games
- Betting lines include spread, over/under, and moneyline
- Adjusted efficiency ratings are key predictors
- Four factors correlate strongly with tournament success
