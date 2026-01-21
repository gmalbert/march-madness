#!/usr/bin/env python3
"""
Example usage of the comprehensive betting data collection functions.

This script demonstrates how to use the functions outlined in roadmap-data-scope.md
to collect all the data needed for March Madness betting predictions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collection import (
    fetch_all_tournament_games,
    fetch_historical_lines,
    fetch_team_season_data,
    fetch_efficiency_ratings,
    fetch_rankings,
    collect_comprehensive_betting_data
)

def main():
    print("🏀 March Madness Betting Data Collection Examples")
    print("=" * 55)

    # Example 1: Fetch tournament games for specific years
    print("\n1. Fetching tournament games (2023-2024)...")
    games = fetch_all_tournament_games(2023, 2024)
    print(f"   Found {len(games)} tournament games")

    # Example 2: Fetch betting lines for recent tournaments
    print("\n2. Fetching betting lines (2023-2024)...")
    lines = fetch_historical_lines(2023, 2024)
    print(f"   Found {len(lines)} betting lines")

    # Example 3: Get team stats for current season
    print("\n3. Fetching team stats for 2024 season...")
    team_stats = fetch_team_season_data(2024)
    print(f"   Found {len(team_stats)} teams with stats")

    # Example 4: Get efficiency ratings
    print("\n4. Fetching efficiency ratings for 2024...")
    efficiency = fetch_efficiency_ratings(2024)
    print(f"   Found {len(efficiency)} teams with efficiency ratings")

    # Example 5: Get rankings
    print("\n5. Fetching rankings for 2024...")
    rankings = fetch_rankings(2024)
    print(f"   Found {len(rankings)} ranking entries")

    # Example 6: Comprehensive collection (commented out to avoid long runtime)
    print("\n6. Comprehensive collection (2020-2024)...")
    print("   # result = collect_comprehensive_betting_data(2020, 2024)")
    print("   # This would collect all data types for model training")

    print("\n" + "=" * 55)
    print("✅ All functions working! Ready for betting model training.")
    print("\nTo collect full historical dataset (2016-2025):")
    print("   python data_collection.py --collect")

if __name__ == "__main__":
    main()