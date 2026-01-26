"""
Odds API Integration for March Madness

This module integrates with The Odds API to provide live betting lines
and enable line movement tracking for NCAA Basketball games.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import time

# Import the opening line database
from opening_line_database import OpeningLineDatabase

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ODDS_API_KEY = os.environ.get("ODDS_API_KEY")
if not ODDS_API_KEY:
    raise ValueError("ODDS_API_KEY not found. Get one from https://the-odds-api.com/")

# Configuration
BASE_URL = "https://api.the-odds-api.com/v4"
DATA_DIR = Path("data_files")
ODDS_CACHE_DIR = DATA_DIR / "odds_cache"
ODDS_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Initialize opening line database
opening_db = OpeningLineDatabase()


class OddsAPIClient:
    """Client for The Odds API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()

    def get_ncaab_odds(self, regions: str = "us", markets: str = "h2h,spreads,totals") -> List[Dict]:
        """Get current NCAA Basketball odds."""
        url = f"{BASE_URL}/sports/basketball_ncaab/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american"
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def get_historical_odds(self, sport: str, date: str, regions: str = "us", markets: str = "h2h,spreads,totals") -> Dict:
        """Get historical odds snapshot (requires paid plan)."""
        url = f"{BASE_URL}/historical/sports/{sport}/odds"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "date": date
        }

        response = self.session.get(url, params=params)
        response.raise_for_status()
        return response.json()


# Global client instance
odds_client = OddsAPIClient(ODDS_API_KEY)


def cache_odds_snapshot() -> str:
    """Capture and cache current odds snapshot."""
    try:
        odds_data = odds_client.get_ncaab_odds()
        timestamp = datetime.now().isoformat()

        # Save to cache
        cache_file = ODDS_CACHE_DIR / f"odds_snapshot_{int(time.time())}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "data": odds_data
            }, f, indent=2)

        print(f"Cached {len(odds_data)} games at {timestamp}")
        return str(cache_file)

    except Exception as e:
        print(f"Error caching odds: {e}")
        return None


def load_cached_odds() -> List[Dict]:
    """Load most recent cached odds."""
    cache_files = list(ODDS_CACHE_DIR.glob("odds_snapshot_*.json"))
    if not cache_files:
        return []

    # Get most recent file
    latest_file = max(cache_files, key=lambda f: f.stat().st_mtime)

    with open(latest_file, 'r') as f:
        cached_data = json.load(f)

    return cached_data.get("data", [])


def find_game_odds(game_teams: Tuple[str, str], odds_data: List[Dict] = None) -> Optional[Dict]:
    """Find odds for a specific game."""
    if odds_data is None:
        odds_data = load_cached_odds()

    # Normalize team names for matching
    team1, team2 = game_teams
    team1_norm = team1.lower().replace("state", "st").replace("university", "").strip()
    team2_norm = team2.lower().replace("state", "st").replace("university", "").strip()

    for game in odds_data:
        home_team = game.get("home_team", "").lower().replace("state", "st").replace("university", "").strip()
        away_team = game.get("away_team", "").lower().replace("state", "st").replace("university", "").strip()

        # Check both possible team orderings
        if ((home_team == team1_norm and away_team == team2_norm) or
            (home_team == team2_norm and away_team == team1_norm)):
            return game

    return None


def extract_line_data(game_odds: Dict) -> Dict:
    """Extract spread, total, and moneyline from game odds."""
    if not game_odds.get("bookmakers"):
        return {}

    # Use first available bookmaker (typically FanDuel or DraftKings)
    bookmaker = game_odds["bookmakers"][0]

    result = {
        "home_team": game_odds.get("home_team"),
        "away_team": game_odds.get("away_team"),
        "commence_time": game_odds.get("commence_time"),
        "bookmaker": bookmaker.get("title"),
        "last_update": bookmaker.get("last_update")
    }

    # Extract markets
    for market in bookmaker.get("markets", []):
        market_key = market["key"]
        outcomes = market.get("outcomes", [])

        if market_key == "spreads":
            # Find spread data
            for outcome in outcomes:
                if outcome["name"] == game_odds["home_team"]:
                    result["home_spread"] = outcome.get("point", 0)
                    result["home_spread_odds"] = outcome.get("price")
                elif outcome["name"] == game_odds["away_team"]:
                    result["away_spread"] = outcome.get("point", 0)
                    result["away_spread_odds"] = outcome.get("price")

        elif market_key == "totals":
            # Find total data
            for outcome in outcomes:
                if outcome["name"] == "Over":
                    result["total_line"] = outcome.get("point")
                    result["over_odds"] = outcome.get("price")
                elif outcome["name"] == "Under":
                    result["under_odds"] = outcome.get("price")

        elif market_key == "h2h":
            # Find moneyline data
            for outcome in outcomes:
                if outcome["name"] == game_odds["home_team"]:
                    result["home_moneyline"] = outcome.get("price")
                elif outcome["name"] == game_odds["away_team"]:
                    result["away_moneyline"] = outcome.get("price")

    return result


def get_current_lines(game_teams: Tuple[str, str]) -> Optional[Dict]:
    """Get current betting lines for a game."""
    game_odds = find_game_odds(game_teams)
    if not game_odds:
        return None

    return extract_line_data(game_odds)


def get_opening_lines(game_teams: Tuple[str, str]) -> Optional[Dict]:
    """
    Get opening lines for a game from the opening line database.

    This function first tries to find exact matches in the database,
    then falls back to similar games if available.
    """
    team1, team2 = game_teams

    # Try to find exact match (this would require game IDs in a real implementation)
    # For now, we'll look for similar games based on team names

    similar_games = opening_db.find_similar_games(team1, team2, limit=1)

    if similar_games:
        game_data = similar_games[0]
        return {
            "home_team": game_data["home_team"],
            "away_team": game_data["away_team"],
            "home_spread": game_data["opening_spread"],
            "total_line": game_data["opening_total"],
            "line_type": "opening",
            "source": "database",
            "game_date": game_data["game_date"],
            "season": game_data["season"]
        }

    # Fallback: Use current lines as proxy for opening lines
    # (This simulates what would happen with limited historical data)
    current = get_current_lines(game_teams)
    if current:
        opening = current.copy()
        opening["line_type"] = "opening (estimated)"
        opening["source"] = "current_lines_fallback"
        return opening

    return None


def calculate_line_movement(current_lines: Dict, opening_lines: Dict) -> Dict:
    """Calculate line movement between opening and current lines."""
    if not current_lines or not opening_lines:
        return {"error": "Missing line data"}

    movement = {
        "spread_movement": 0,
        "total_movement": 0,
        "sharp_money_indicator": "No significant movement"
    }

    # Calculate spread movement
    if (current_lines.get("home_spread") is not None and
        opening_lines.get("home_spread") is not None):
        spread_diff = current_lines["home_spread"] - opening_lines["home_spread"]
        movement["spread_movement"] = spread_diff

        # Analyze sharp money based on spread movement
        if abs(spread_diff) >= 3:
            if spread_diff > 0:
                movement["sharp_money_indicator"] = "Heavy money on home team"
            else:
                movement["sharp_money_indicator"] = "Heavy money on away team"
        elif abs(spread_diff) >= 1:
            movement["sharp_money_indicator"] = "Moderate line movement"

    # Calculate total movement
    if (current_lines.get("total_line") is not None and
        opening_lines.get("total_line") is not None):
        total_diff = current_lines["total_line"] - opening_lines["total_line"]
        movement["total_movement"] = total_diff

    return movement


def track_line_movement(game_teams: Tuple[str, str]) -> Dict:
    """Track line movement for a game."""
    current = get_current_lines(game_teams)
    opening = get_opening_lines(game_teams)

    # If no specific game found, return first available game for demonstration
    if not current:
        odds_data = load_cached_odds()
        if odds_data:
            first_game = odds_data[0]
            game_teams = (first_game["away_team"], first_game["home_team"])
            current = get_current_lines(game_teams)
            opening = get_opening_lines(game_teams)

    if not current:
        return {"error": f"No current odds found for {game_teams[0]} vs {game_teams[1]}"}

    result = {
        "game_teams": game_teams,
        "current_lines": current,
        "opening_lines": opening,
        "line_movement": calculate_line_movement(current, opening) if opening else {"error": "No opening lines available"}
    }

    return result


def populate_opening_lines_from_sportsdataio(season: str = "2023"):
    """
    Populate opening lines database with data from SportsDataIO.

    This function would be called after signing up for SportsDataIO free trial.
    For now, it shows the structure for integration.
    """
    print(f"Populating opening lines for season {season}...")
    print("Note: This requires SportsDataIO API key and free trial signup")
    print("Visit: https://sportsdata.io/ for College Basketball API")

    # Example structure for SportsDataIO integration:
    # 1. Get games for the season
    # 2. For each game, get historical odds data
    # 3. Extract opening lines and store in database

    # Placeholder for actual implementation
    print("SportsDataIO integration would go here...")
    print("Example API call: GET https://api.sportsdata.io/v2/json/GamesBySeason/2023")
    print("Then for each game: GET https://api.sportsdata.io/v2/json/HistoricalOddsByGame/{game_id}")


def populate_opening_lines_manually():
    """
    Manually populate some sample opening lines for demonstration.
    In practice, this would be automated from API sources.
    """
    sample_games = [
        {
            "game_id": "2024_001",
            "home_team": "Duke",
            "away_team": "North Carolina",
            "opening_spread": -3.5,
            "opening_total": 145.5,
            "game_date": "2024-01-15",
            "season": "2023-24"
        },
        {
            "game_id": "2024_002",
            "home_team": "Kansas",
            "away_team": "Texas",
            "opening_spread": -2.0,
            "opening_total": 138.0,
            "game_date": "2024-01-20",
            "season": "2023-24"
        },
        {
            "game_id": "2024_003",
            "home_team": "UConn",
            "away_team": "Creighton",
            "opening_spread": -4.0,
            "opening_total": 142.5,
            "game_date": "2024-02-10",
            "season": "2023-24"
        },
        {
            "game_id": "2024_004",
            "home_team": "Arizona",
            "away_team": "UCLA",
            "opening_spread": -1.5,
            "opening_total": 149.0,
            "game_date": "2024-02-15",
            "season": "2023-24"
        }
    ]

    for game in sample_games:
        opening_db.add_opening_line(**game)

    print(f"Added {len(sample_games)} sample opening lines to database")


def demo_line_movement_tracking():
    """Demonstrate line movement tracking with current live odds and opening line database."""
    print("=== Odds API Line Movement Tracking Demo ===\n")

    # First populate some sample opening lines
    print("Populating sample opening lines...")
    populate_opening_lines_manually()
    print()

    # Cache fresh odds
    print("Fetching fresh odds from The Odds API...")
    cache_file = cache_odds_snapshot()
    if not cache_file:
        print("Failed to cache odds")
        return

    # Load cached odds
    odds_data = load_cached_odds()
    print(f"Loaded {len(odds_data)} games from cache\n")

    # Show line movement analysis for first few games
    games_analyzed = 0
    for game in odds_data[:5]:  # Check more games to find matches
        teams = (game["home_team"], game["away_team"])
        movement_data = track_line_movement(teams)

        # Only show games where we have both current and opening data
        current = movement_data.get("current_lines", {})
        opening = movement_data.get("opening_lines")

        if current and opening and opening.get("source") != "current_lines_fallback":
            games_analyzed += 1
            print(f"Game {games_analyzed}: {teams[0]} vs {teams[1]}")
            print(f"Commence: {current.get('commence_time', 'Unknown')}")

            if current.get("home_spread"):
                print(f"Current Spread: {current['home_spread']} (home)")
            if current.get("total_line"):
                print(f"Current Total: {current['total_line']}")

            if opening.get("home_spread"):
                print(f"Opening Spread: {opening['home_spread']} (home)")
            if opening.get("total_line"):
                print(f"Opening Total: {opening['total_line']}")

            movement = movement_data.get("line_movement", {})
            if "error" not in movement:
                print(f"Line Movement: Spread {movement.get('spread_movement', 0)}, Total {movement.get('total_movement', 0)}")
                print(f"Analysis: {movement.get('sharp_money_indicator', 'Unknown')}")
            else:
                print("Line Movement: Unable to calculate")

            print()

            if games_analyzed >= 3:  # Show up to 3 games
                break

    if games_analyzed == 0:
        print("No games found with both current odds and opening lines.")
        print("This could be because:")
        print("1. Team names don't match between live odds and database")
        print("2. No opening lines in database for these teams")
        print("3. Using fallback opening lines (same as current)")
        print("\nShowing first available game with current odds:")

        if odds_data:
            first_game = odds_data[0]
            teams = (first_game["home_team"], first_game["away_team"])
            movement_data = track_line_movement(teams)

            current = movement_data.get("current_lines", {})
            print(f"Game: {teams[0]} vs {teams[1]}")
            print(f"Commence: {current.get('commence_time', 'Unknown')}")

            if current.get("home_spread"):
                print(f"Current Spread: {current['home_spread']} (home)")
            if current.get("total_line"):
                print(f"Current Total: {current['total_line']}")

            opening = movement_data.get("opening_lines")
            if opening and opening.get("source") == "current_lines_fallback":
                print("Opening lines: Using current lines as estimate (no historical data)")
            else:
                print("Opening lines: Not available")

    # Show database stats
    print(f"\nOpening Lines Database Status:")
    stats = opening_db.get_season_stats("2023-24")
    print(f"Games in database: {stats['total_games']}")
    if stats['total_games'] > 0:
        print(f"Average spread: {stats['avg_spread']:.1f}")
        print(f"Average total: {stats['avg_total']:.1f}")

    print(f"\nTo get real historical data:")
    print("1. Sign up for SportsDataIO free trial: https://sportsdata.io/")
    print("2. Use their College Basketball API for historical odds")
    print("3. Call populate_opening_lines_from_sportsdataio() with real data")


if __name__ == "__main__":
    demo_line_movement_tracking()