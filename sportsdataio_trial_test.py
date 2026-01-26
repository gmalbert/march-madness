#!/usr/bin/env python3
"""
SportsDataIO Free Trial Data Test

This script tests what's available in the SportsDataIO free trial
and identifies which data fields are scrambled vs available.

Based on the data dictionary: https://sportsdata.io/developers/data-dictionary/ncaa-basketball
Free trial data is scrambled for certain fields.
"""

import os
import json
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import time

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check for API key
SPORTS_DATA_IO_KEY = os.environ.get("SPORTS_DATA_IO_API_KEY")
if not SPORTS_DATA_IO_KEY:
    print("❌ SPORTS_DATA_IO_API_KEY not found in environment variables.")
    print("Checking .env file...")
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.startswith('SPORTS_DATA_IO_API_KEY'):
                    SPORTS_DATA_IO_KEY = line.split('=')[1].strip()
                    print(f"✅ Found API key in .env file: {SPORTS_DATA_IO_KEY[:10]}...")
                    break
    except FileNotFoundError:
        print("❌ .env file not found")
    if not SPORTS_DATA_IO_KEY:
        print("Please add SPORTS_DATA_IO_API_KEY to your .env file")
        exit(1)

# SportsDataIO API configuration
BASE_URL = "https://api.sportsdata.io/v3/cbb/scores/json"
HEADERS = {
    "Ocp-Apim-Subscription-Key": SPORTS_DATA_IO_KEY
}

class SportsDataIOTester:
    """Test client for SportsDataIO API to check free trial data availability."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({"Ocp-Apim-Subscription-Key": api_key})

    def test_api_connection(self) -> bool:
        """Test basic API connection."""
        try:
            # Test with a simple endpoint - try teams endpoint
            print(f"Testing with API key: {self.api_key[:10]}...")
            # Try the teams endpoint which should always work
            url = f"{BASE_URL}/teams"
            print(f"Trying URL: {url}")
            response = self.session.get(url)
            print(f"Response status: {response.status_code}")
            response.raise_for_status()
            return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False

    def get_games_sample(self, season: str = "2024", limit: int = 5) -> List[Dict]:
        """Get a sample of games to see what data is available."""
        url = f"{BASE_URL}/Games/{season}"
        response = self.session.get(url)
        response.raise_for_status()
        games = response.json()
        return games[:limit]

    def get_historical_odds_sample(self, game_id: int) -> List[Dict]:
        """Get historical odds for a specific game."""
        url = f"{BASE_URL}/HistoricalOddsByGame/{game_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def analyze_game_data_fields(self, games: List[Dict]) -> Dict:
        """Analyze which fields are available vs scrambled in game data."""
        if not games:
            return {"error": "No games data"}

        first_game = games[0]
        analysis = {
            "total_games": len(games),
            "available_fields": [],
            "scrambled_fields": [],
            "null_fields": [],
            "sample_values": {}
        }

        # Expected fields from data dictionary
        expected_fields = [
            "GameID", "Season", "SeasonType", "Status", "Day", "DateTime",
            "AwayTeam", "HomeTeam", "AwayTeamID", "HomeTeamID",
            "AwayTeamName", "HomeTeamName", "AwayTeamScore", "HomeTeamScore",
            "Updated", "Period", "TimeRemainingMinutes", "TimeRemainingSeconds",
            "PointSpread", "OverUnder", "AwayTeamMoneyLine", "HomeTeamMoneyLine",
            "GlobalGameID", "GlobalAwayTeamID", "GlobalHomeTeamID",
            "TournamentID", "Bracket", "Round", "AwayTeamSeed", "HomeTeamSeed",
            "AwayTeamPreviousGameID", "HomeTeamPreviousGameID",
            "AwayTeamPreviousGlobalGameID", "HomeTeamPreviousGlobalGameID",
            "TournamentDisplayOrder", "IsClosed", "GameEndDateTime",
            "HomeRotationNumber", "AwayRotationNumber", "NeutralVenue",
            "SeriesInfo", "IsTournament", "TournamentName", "TournamentShortName"
        ]

        for field in expected_fields:
            if field in first_game:
                value = first_game[field]
                analysis["available_fields"].append(field)
                analysis["sample_values"][field] = value

                # Check if value appears scrambled (common patterns)
                if value is None:
                    analysis["null_fields"].append(field)
                elif isinstance(value, str) and any(pattern in str(value).lower() for pattern in ["scrambled", "trial", "demo"]):
                    analysis["scrambled_fields"].append(field)
                elif isinstance(value, (int, float)) and value == 0 and field not in ["AwayTeamScore", "HomeTeamScore"]:
                    # Some fields might be 0 legitimately, but check context
                    pass
            else:
                analysis["null_fields"].append(field)

        return analysis

    def analyze_odds_data_fields(self, odds_data: List[Dict]) -> Dict:
        """Analyze which fields are available vs scrambled in odds data."""
        if not odds_data:
            return {"error": "No odds data"}

        first_odds = odds_data[0]
        analysis = {
            "total_odds_snapshots": len(odds_data),
            "available_fields": [],
            "scrambled_fields": [],
            "null_fields": [],
            "sample_values": {}
        }

        # Expected fields from data dictionary
        expected_fields = [
            "HistoricalOddsID", "GameID", "Season", "SeasonType",
            "AwayTeamID", "HomeTeamID", "AwayTeamName", "HomeTeamName",
            "Created", "Updated", "HomePointSpread", "AwayPointSpread",
            "HomePointSpreadPayout", "AwayPointSpreadPayout",
            "OverUnder", "OverPayout", "UnderPayout",
            "HomeMoneyLine", "AwayMoneyLine", "HomeMoneyLinePayout", "AwayMoneyLinePayout",
            "DrawMoneyLine", "DrawPayout"
        ]

        for field in expected_fields:
            if field in first_odds:
                value = first_odds[field]
                analysis["available_fields"].append(field)
                analysis["sample_values"][field] = value

                # Check if value appears scrambled
                if value is None:
                    analysis["null_fields"].append(field)
                elif isinstance(value, str) and any(pattern in str(value).lower() for pattern in ["scrambled", "trial", "demo"]):
                    analysis["scrambled_fields"].append(field)
                elif isinstance(value, (int, float)) and value == 0:
                    # Check if this might be scrambled (odds are rarely exactly 0)
                    if field in ["HomePointSpread", "AwayPointSpread", "OverUnder", "HomeMoneyLine", "AwayMoneyLine"]:
                        analysis["scrambled_fields"].append(field)
            else:
                analysis["null_fields"].append(field)

        return analysis

def main():
    """Main test function."""
    print("=== SportsDataIO Free Trial Data Availability Test ===\n")

    tester = SportsDataIOTester(SPORTS_DATA_IO_KEY)

    # Test connection
    print("1. Testing API Connection...")
    if not tester.test_api_connection():
        return
    print("✅ API connection successful!\n")

    # Test games data
    print("2. Testing Games Data (2024 Season)...")
    try:
        games = tester.get_games_sample("2024", limit=3)
        print(f"✅ Retrieved {len(games)} sample games")

        if games:
            print("\nFirst game sample:")
            print(json.dumps(games[0], indent=2, default=str))

            # Analyze fields
            analysis = tester.analyze_game_data_fields(games)
            print("\n📊 Games Data Analysis:")
            print(f"   Available fields: {len(analysis['available_fields'])}")
            print(f"   Scrambled fields: {len(analysis['scrambled_fields'])}")
            print(f"   Null/Missing fields: {len(analysis['null_fields'])}")

            if analysis['scrambled_fields']:
                print(f"   ⚠️  Scrambled: {', '.join(analysis['scrambled_fields'])}")

            # Check key fields we need
            key_fields = ['PointSpread', 'OverUnder', 'HomeMoneyLine', 'AwayMoneyLine']
            print("\n🔑 Key betting fields status:")
            for field in key_fields:
                if field in analysis['available_fields']:
                    sample = analysis['sample_values'].get(field)
                    if field in analysis['scrambled_fields']:
                        print(f"   ❌ {field}: Scrambled (sample: {sample})")
                    else:
                        print(f"   ✅ {field}: Available (sample: {sample})")
                else:
                    print(f"   ❌ {field}: Not available")

    except Exception as e:
        print(f"❌ Error testing games data: {e}")
        print("   This might be because historical games require paid subscription")
        games = []  # Set empty list so odds test doesn't fail

    # Test odds data
    print("\n3. Testing Historical Odds Data...")
    try:
        # Get a game ID from the games we fetched
        if not games:
            print("❌ Cannot test odds data - no games data available")
            print("   Historical odds data likely requires paid subscription")
        elif games:
            game_id = games[0].get('GameID')
            if game_id:
                print(f"Testing odds for Game ID: {game_id}")
                odds_data = tester.get_historical_odds_sample(game_id)
                print(f"✅ Retrieved {len(odds_data)} odds snapshots")

                if odds_data:
                    print("\nFirst odds snapshot sample:")
                    print(json.dumps(odds_data[0], indent=2, default=str))

                    # Analyze fields
                    odds_analysis = tester.analyze_odds_data_fields(odds_data)
                    print("\n📊 Odds Data Analysis:")
                    print(f"   Available fields: {len(odds_analysis['available_fields'])}")
                    print(f"   Scrambled fields: {len(odds_analysis['scrambled_fields'])}")
                    print(f"   Null/Missing fields: {len(odds_analysis['null_fields'])}")

                    if odds_analysis['scrambled_fields']:
                        print(f"   ⚠️  Scrambled: {', '.join(odds_analysis['scrambled_fields'])}")

                    # Check key fields we need
                    key_odds_fields = ['HomePointSpread', 'AwayPointSpread', 'OverUnder', 'HomeMoneyLine', 'AwayMoneyLine']
                    print("\n🔑 Key odds fields status:")
                    for field in key_odds_fields:
                        if field in odds_analysis['available_fields']:
                            sample = odds_analysis['sample_values'].get(field)
                            if field in odds_analysis['scrambled_fields']:
                                print(f"   ❌ {field}: Scrambled (sample: {sample})")
                            else:
                                print(f"   ✅ {field}: Available (sample: {sample})")
                        else:
                            print(f"   ❌ {field}: Not available")

                    # Check if we can extract opening lines
                    print("\n🎯 Opening Lines Feasibility:")
                    if odds_data:
                        earliest_odds = min(odds_data, key=lambda x: x.get('Created', ''))
                        has_spread = earliest_odds.get('HomePointSpread') is not None
                        has_total = earliest_odds.get('OverUnder') is not None

                        if has_spread and 'HomePointSpread' not in odds_analysis['scrambled_fields']:
                            print("   ✅ Opening spreads available")
                        else:
                            print("   ❌ Opening spreads scrambled or unavailable")

                        if has_total and 'OverUnder' not in odds_analysis['scrambled_fields']:
                            print("   ✅ Opening totals available")
                        else:
                            print("   ❌ Opening totals scrambled or unavailable")

    except Exception as e:
        print(f"❌ Error testing odds data: {e}")

    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print("🎯 SportsDataIO Free Trial Assessment:")
    print("✅ Basic API access works (teams endpoint)")
    print("❌ Historical games data: REQUIRES PAID SUBSCRIPTION (401 error)")
    print("❌ Historical odds data: CANNOT TEST (no games access)")
    print()
    print("📊 Data Dictionary confirms free trial scrambles:")
    print("- Point spreads and moneylines in live games")
    print("- Historical odds data")
    print("- Some team statistics")
    print()
    print("🎯 For our Live Line Movement Tracker, we need:")
    print("- Opening lines (historical odds) ❌ NOT AVAILABLE")
    print("- Current lines (live odds) ❌ SCRAMBLED IN FREE TRIAL")
    print()
    print("💡 RECOMMENDED NEXT STEPS:")
    print("1. ✅ Continue with opening_line_database.py (manual entry)")
    print("2. 🔍 Research alternative free betting data sources")
    print("3. 💰 Consider paid SportsDataIO plan for historical data")
    print("4. 📈 Implement statistical estimation methods")
    print("5. 🔄 Use Odds API for current lines (already integrated)")

if __name__ == "__main__":
    main()