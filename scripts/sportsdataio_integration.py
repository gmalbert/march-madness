#!/usr/bin/env python3
"""
SportsDataIO Integration Script

This script integrates with SportsDataIO to populate the opening lines database
with historical college basketball betting data.

SportsDataIO offers:
- Free trial with access to historical odds from 2019+
- College Basketball API with game odds and line movement data
- Opening lines, current lines, and historical data

To use this script:
1. Sign up for free trial at https://sportsdata.io/
2. Get your API key from the dashboard
3. Set SPORTS_DATA_IO_KEY environment variable
4. Run this script to populate historical opening lines
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

# Import our opening line database
from opening_line_database import OpeningLineDatabase

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SPORTS_DATA_IO_KEY = os.environ.get("SPORTS_DATA_IO_KEY")
if not SPORTS_DATA_IO_KEY:
    print("SPORTS_DATA_IO_KEY not found in environment variables.")
    print("Please sign up for SportsDataIO free trial at https://sportsdata.io/")
    print("Then set SPORTS_DATA_IO_KEY in your .env file")
    exit(1)

# SportsDataIO API configuration
BASE_URL = "https://api.sportsdata.io/v2/json"
HEADERS = {
    "Ocp-Apim-Subscription-Key": SPORTS_DATA_IO_KEY
}

# Initialize database
db = OpeningLineDatabase()

class SportsDataIOClient:
    """Client for SportsDataIO API."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Ocp-Apim-Subscription-Key": api_key})

    def get_games_by_season(self, season: str) -> List[Dict]:
        """Get all games for a season."""
        url = f"{BASE_URL}/GamesBySeason/{season}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_historical_odds_by_game(self, game_id: int) -> List[Dict]:
        """Get historical odds for a specific game."""
        url = f"{BASE_URL}/HistoricalOddsByGame/{game_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

# Global client
sports_client = SportsDataIOClient(SPORTS_DATA_IO_KEY)

def extract_opening_lines(odds_data: List[Dict]) -> Optional[Dict]:
    """
    Extract opening lines from SportsDataIO odds data.

    SportsDataIO provides odds snapshots over time. The opening line
    is typically the first odds posted for the game.
    """
    if not odds_data:
        return None

    # Sort by date to find the earliest odds (opening line)
    sorted_odds = sorted(odds_data, key=lambda x: x.get('Created', ''))

    # Get the first (earliest) odds entry
    opening_odds = sorted_odds[0]

    # Extract spread and total
    spread = opening_odds.get('HomePointSpread')
    total = opening_odds.get('OverUnder')

    if spread is None or total is None:
        return None

    return {
        'spread': spread,
        'total': total,
        'odds_id': opening_odds.get('HistoricalOddsID'),
        'created': opening_odds.get('Created')
    }

def populate_season_data(season: str, max_games: int = 50):
    """
    Populate opening lines for an entire season.

    Args:
        season: Season year (e.g., "2023" for 2022-23 season)
        max_games: Maximum games to process (for testing)
    """
    print(f"Fetching games for season {season}...")

    try:
        games = sports_client.get_games_by_season(season)
        print(f"Found {len(games)} games in season {season}")
    except Exception as e:
        print(f"Error fetching games: {e}")
        return

    games_processed = 0
    games_with_odds = 0

    for game in games[:max_games]:  # Limit for testing
        game_id = game.get('GameID')
        home_team = game.get('HomeTeamName', game.get('HomeTeam'))
        away_team = game.get('AwayTeamName', game.get('AwayTeam'))
        game_date = game.get('DateTime')

        if not all([game_id, home_team, away_team, game_date]):
            continue

        try:
            # Get historical odds for this game
            odds_data = sports_client.get_historical_odds_by_game(game_id)

            if odds_data:
                opening_lines = extract_opening_lines(odds_data)

                if opening_lines:
                    # Format season string
                    season_str = f"{int(season)-1}-{season[-2:]}"

                    # Add to database
                    db.add_opening_line(
                        game_id=str(game_id),
                        home_team=home_team,
                        away_team=away_team,
                        opening_spread=opening_lines['spread'],
                        opening_total=opening_lines['total'],
                        game_date=game_date.split('T')[0],  # Extract date part
                        season=season_str
                    )

                    games_with_odds += 1

            games_processed += 1

            # Progress update
            if games_processed % 10 == 0:
                print(f"Processed {games_processed} games, {games_with_odds} with odds data...")

            # Rate limiting - SportsDataIO has rate limits
            time.sleep(0.1)

        except Exception as e:
            print(f"Error processing game {game_id}: {e}")
            continue

    print(f"\nCompleted processing {games_processed} games")
    print(f"Found opening lines for {games_with_odds} games")

    # Show season stats
    season_str = f"{int(season)-1}-{season[-2:]}"
    stats = db.get_season_stats(season_str)
    print(f"Season {season_str} stats: {stats}")

def test_api_connection():
    """Test the SportsDataIO API connection."""
    print("Testing SportsDataIO API connection...")

    try:
        # Try to get a small amount of data
        games = sports_client.get_games_by_season("2023")
        print(f"✅ API connection successful! Found {len(games)} games for 2023 season")

        # Test odds for first game
        if games:
            first_game = games[0]
            game_id = first_game.get('GameID')
            print(f"Testing odds retrieval for game {game_id}...")

            odds_data = sports_client.get_historical_odds_by_game(game_id)
            if odds_data:
                print(f"✅ Odds data available! Found {len(odds_data)} odds snapshots")
                opening = extract_opening_lines(odds_data)
                if opening:
                    print(f"✅ Opening lines extracted: Spread {opening['spread']}, Total {opening['total']}")
                else:
                    print("⚠️  Could not extract opening lines from odds data")
            else:
                print("⚠️  No odds data available for this game")

        return True

    except Exception as e:
        print(f"❌ API connection failed: {e}")
        print("Please check your API key and internet connection")
        return False

def main():
    """Main function to run the SportsDataIO integration."""
    print("=== SportsDataIO College Basketball Integration ===\n")

    # Test connection first
    if not test_api_connection():
        return

    print("\n" + "="*50)
    print("Choose an option:")
    print("1. Populate data for 2023 season (2022-23)")
    print("2. Populate data for 2022 season (2021-22)")
    print("3. Test with limited data (first 10 games)")
    print("4. Export current database to CSV")
    print("="*50)

    choice = input("\nEnter your choice (1-4): ").strip()

    if choice == "1":
        populate_season_data("2023")
    elif choice == "2":
        populate_season_data("2022")
    elif choice == "3":
        populate_season_data("2023", max_games=10)
    elif choice == "4":
        db.export_to_csv("sportsdataio_opening_lines.csv")
        print("Database exported to sportsdataio_opening_lines.csv")
    else:
        print("Invalid choice")

    print("\nIntegration complete!")
    print("You can now use the opening lines database with your line movement tracker.")

if __name__ == "__main__":
    main()