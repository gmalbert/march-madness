# Roadmap: Data Gathering

*How to connect to CFBD and pull data for analysis.*

## Prerequisites
1. Install dependencies: `pip install -r requirements.txt`
2. Obtain API key from [CollegeFootballData.com](https://collegefootballdata.com/)
3. Set environment variable or store in `.env` file

## API Configuration

```python
import os
import cfbd
from cfbd.rest import ApiException

# Load from environment or .env file
API_KEY = os.environ.get("CFBD_API_KEY")

# Configure the client
configuration = cfbd.Configuration(
    host="https://api.collegefootballdata.com",
    access_token=API_KEY
)

def get_api_client():
    """Returns a configured API client."""
    return cfbd.ApiClient(configuration)
```

## Basic Data Pull Pattern

```python
def fetch_games(year: int, season_type: str = "regular"):
    """Fetch all games for a given year."""
    with get_api_client() as api_client:
        games_api = cfbd.GamesApi(api_client)
        try:
            games = games_api.get_games(year=year, season_type=season_type)
            return games
        except ApiException as e:
            print(f"Error fetching games: {e}")
            return []
```

## Caching Strategy

To avoid hitting rate limits and speed up development:

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

## Batch Data Collection

```python
def collect_historical_data(start_year: int, end_year: int):
    """Collect games data for multiple years."""
    all_games = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching {year}...")
        games = fetch_games(year)
        all_games.extend(games)
        cache_data(f"games_{year}", games)
    return all_games
```

## Error Handling Best Practices

```python
import time

def fetch_with_retry(fetch_func, max_retries=3, delay=2):
    """Retry API calls on failure."""
    for attempt in range(max_retries):
        try:
            return fetch_func()
        except ApiException as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed, retrying...")
                time.sleep(delay * (attempt + 1))
            else:
                raise e
```

## Next Steps
- See `roadmap-data-scope.md` for what data to collect
- See `roadmap-features.md` for feature engineering ideas
