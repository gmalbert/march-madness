# CFBD Python Library Notes

*These notes are prepared for Copilot instructions to assist the agent in reviewing the CFBD library when questions arise.*

## Overview
The CFBD (College Football Data) Python library is an automatically generated wrapper for the College Football Data API. It provides access to a wide range of college football datasets and analytics, including games, teams, players, stats, rankings, and more. The library is maintained by CFBD and is based on OpenAPI specifications.

- **API Version**: 5.13.2
- **Package Version**: 5.13.2
- **Generator**: OpenAPI Generator (Python Pydantic V1 Client)
- **Source**: https://github.com/CFBD/cfbd-python

## Requirements
- Python 3.7 or higher

## Installation
Install the library using pip:
```
pip install cfbd
```
Note: You may need root permissions: `sudo pip install cfbd`

## Authentication
To use the API, you need an API key from [CollegeFootballData.com](https://collegefootballdata.com/). The authentication uses Bearer token authorization.

Example configuration:
```python
import cfbd

configuration = cfbd.Configuration(
    host="https://api.collegefootballdata.com"
)

# Configure Bearer authorization
configuration = cfbd.Configuration(
    access_token=os.environ["BEARER_TOKEN"]
)
```

## Key Features and API Endpoints
The library provides access to numerous endpoints categorized by functionality:

### Games
- `get_games`: Retrieve game data
- `get_scoreboard`: Current scoreboard
- `get_advanced_box_score`: Advanced box scores
- `get_game_player_stats`: Player stats for games
- `get_game_team_stats`: Team stats for games
- `get_media`: Game media
- `get_weather`: Game weather conditions
- `get_records`: Team records

### Teams
- `get_teams`: List of teams
- `get_fbs_teams`: FBS teams
- `get_roster`: Team rosters
- `get_talent`: Team talent rankings
- `get_teams_ats`: Against the spread records
- `get_matchup`: Team matchup history

### Players
- `search_players`: Search for players
- `get_player_usage`: Player usage stats
- `get_returning_production`: Returning production
- `get_transfer_portal`: Transfer portal data

### Stats and Metrics
- `get_advanced_season_stats`: Advanced season stats
- `get_player_season_stats`: Player season stats
- `get_team_stats`: Team season stats
- `get_predicted_points_added_by_player_season`: PPA by player
- `get_predicted_points_added_by_team`: PPA by team
- `get_win_probability`: Win probability
- `get_pregame_win_probabilities`: Pregame win probabilities

### Rankings and Ratings
- `get_rankings`: Poll rankings
- `get_elo`: Elo ratings
- `get_fpi`: Football Power Index
- `get_sp`: S&P+ ratings
- `get_srs`: Simple Rating System

### Recruiting
- `get_recruits`: Recruit data
- `get_team_recruiting_rankings`: Team recruiting rankings
- `get_aggregated_team_recruiting_ratings`: Aggregated recruiting

### Other
- `get_lines`: Betting lines
- `get_coaches`: Coach information
- `get_conferences`: Conference data
- `get_draft_picks`: NFL draft picks
- `get_plays`: Play-by-play data
- `get_drives`: Drive data
- `get_venues`: Venue information

## Data Models
The library includes comprehensive Pydantic models for all response data, such as:
- Game, Team, Player, Coach
- Advanced stats (EPA, PPA, etc.)
- Rankings, Ratings
- Recruiting data
- Betting information

## Basic Usage Example
```python
import cfbd
from cfbd.rest import ApiException

# Configure API client
configuration = cfbd.Configuration(access_token="your_api_key_here")
api_client = cfbd.ApiClient(configuration)

# Example: Get games for a specific year
games_api = cfbd.GamesApi(api_client)
try:
    games = games_api.get_games(year=2023)
    for game in games:
        print(f"{game.home_team} vs {game.away_team}: {game.home_points}-{game.away_points}")
except ApiException as e:
    print(f"Exception: {e}")
```

## Notes
- All URIs are relative to https://api.collegefootballdata.com
- The library uses Pydantic for data validation and serialization
- Extensive documentation for models and endpoints is available in the GitHub repo's docs folder
- This is specifically for college football data, not basketball (note: the project is March Madness, which is basketball, so this may be intended for football analytics or a similar purpose)

*These notes are prepared for Copilot instructions to assist the agent in reviewing the CFBD library when questions arise.*