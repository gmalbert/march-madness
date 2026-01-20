# Roadmap: Data Gathering (Basketball Betting)

*How to connect to CBBD and pull basketball data for betting predictions.*

## Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Obtain API key from [CollegeBasketballData.com](https://collegebasketballdata.com/)
3. Set environment variable `CBBD_API_KEY` or store in `.env` file

## API Configuration

```python
import os
import cbbd
from cbbd.rest import ApiException

API_KEY = os.environ.get("CBBD_API_KEY")

configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com",
    access_token=API_KEY
)

def get_api_client():
    """Returns a configured API client."""
    return cbbd.ApiClient(configuration)
```

## Fetching Games Data

```python
def fetch_games(year: int, season_type: str = "regular"):
    """Fetch games for a given year and season type.
    
    season_type options: 'regular', 'postseason'
    Use 'postseason' for March Madness tournament games.
    """
    with get_api_client() as api_client:
        games_api = cbbd.GamesApi(api_client)
        try:
            games = games_api.get_games(year=year, season_type=season_type)
            return games
        except ApiException as e:
            print(f"Error fetching games: {e}")
            return []

def fetch_tournament_games(year: int):
    """Fetch March Madness tournament games specifically."""
    return fetch_games(year, season_type="postseason")
```

## Fetching Betting Lines (Critical)

```python
def fetch_betting_lines(year: int, season_type: str = "postseason"):
    """Fetch betting lines including spreads, over/unders, moneylines."""
    with get_api_client() as api_client:
        lines_api = cbbd.LinesApi(api_client)
        try:
            lines = lines_api.get_lines(year=year, season_type=season_type)
            return lines
        except ApiException as e:
            print(f"Error fetching lines: {e}")
            return []

def fetch_line_providers():
    """Get list of available betting line providers."""
    with get_api_client() as api_client:
        lines_api = cbbd.LinesApi(api_client)
        return lines_api.get_providers()
```

## Fetching Team Statistics

```python
def fetch_team_stats(year: int):
    """Fetch team season statistics."""
    with get_api_client() as api_client:
        stats_api = cbbd.StatsApi(api_client)
        return stats_api.get_team_season_stats(year=year)

def fetch_team_shooting_stats(year: int):
    """Fetch detailed shooting statistics."""
    with get_api_client() as api_client:
        stats_api = cbbd.StatsApi(api_client)
        return stats_api.get_team_season_shooting_stats(year=year)
```

## Fetching Efficiency Ratings

```python
def fetch_adjusted_efficiency(year: int):
    """Fetch adjusted efficiency ratings (KenPom-style metrics)."""
    with get_api_client() as api_client:
        ratings_api = cbbd.RatingsApi(api_client)
        return ratings_api.get_adjusted_efficiency(year=year)

def fetch_srs_ratings(year: int):
    """Fetch Simple Rating System ratings."""
    with get_api_client() as api_client:
        ratings_api = cbbd.RatingsApi(api_client)
        return ratings_api.get_srs(year=year)
```

## Caching Strategy

```python
import json
from pathlib import Path

CACHE_DIR = Path("data_files/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def cache_data(filename: str, data: list):
    """Save data to cache file."""
    filepath = CACHE_DIR / f"{filename}.json"
    with open(filepath, "w") as f:
        json.dump([item.to_dict() for item in data], f, indent=2, default=str)

def load_cached(filename: str):
    """Load data from cache if exists."""
    filepath = CACHE_DIR / f"{filename}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    return None
```

## Batch Collection for Training Data

```python
def collect_historical_betting_data(start_year: int, end_year: int):
    """Collect all data needed for betting model training."""
    for year in range(start_year, end_year + 1):
        print(f"Collecting {year} data...")
        
        # Core data
        games = fetch_games(year)
        cache_data(f"games_{year}", games)
        
        tournament = fetch_tournament_games(year)
        cache_data(f"tournament_{year}", tournament)
        
        # Betting lines
        lines = fetch_betting_lines(year, "postseason")
        cache_data(f"lines_tournament_{year}", lines)
        
        # Team stats
        stats = fetch_team_stats(year)
        cache_data(f"team_stats_{year}", stats)
        
        # Ratings
        efficiency = fetch_adjusted_efficiency(year)
        cache_data(f"efficiency_{year}", efficiency)
        
        print(f"  {year} complete")
```

## Next Steps
- See `roadmap-data-scope.md` for what data to collect
- See `roadmap-betting-features.md` for betting-specific features
