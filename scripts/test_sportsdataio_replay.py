#!/usr/bin/env python3
"""
Test SportsDataIO Replay API

The Replay API provides historical data playback - simulating real-time data
from past seasons. This could be useful for backtesting and model training.

Example: https://sportsdata.io/developers/replay/cbb-2023-regular-season-november-29
"""

import os
import json
import requests
from datetime import datetime

# Load API key
try:
    with open('.env', 'r') as f:
        for line in f:
            if line.startswith('SPORTS_DATA_IO_API_KEY'):
                SPORTS_DATA_IO_KEY = line.split('=')[1].strip()
                break
except FileNotFoundError:
    print("❌ .env file not found")
    exit(1)

# API configuration
BASE_URL = "https://api.sportsdata.io"
session = requests.Session()
session.headers.update({"Ocp-Apim-Subscription-Key": SPORTS_DATA_IO_KEY})

def test_replay_endpoint(url, description):
    """Test a replay endpoint."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"URL: {url}")
    try:
        response = session.get(url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"✅ SUCCESS - Retrieved {len(data)} items")
                if data:
                    print(f"\nFirst item:")
                    print(json.dumps(data[0], indent=2, default=str)[:800] + "...")
                return True, data
            elif isinstance(data, dict):
                print(f"✅ SUCCESS - Retrieved data object")
                print(f"\nKeys: {list(data.keys())}")
                print(f"\nSample:")
                print(json.dumps(data, indent=2, default=str)[:800] + "...")
                return True, data
            else:
                print(f"✅ SUCCESS - Retrieved: {data}")
                return True, data
        elif response.status_code == 401:
            print(f"❌ UNAUTHORIZED - Requires paid subscription")
            return False, None
        elif response.status_code == 403:
            print(f"❌ FORBIDDEN - Not available in current plan")
            return False, None
        elif response.status_code == 404:
            print(f"❌ NOT FOUND - Endpoint doesn't exist or data not available")
            return False, None
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            print(f"Response: {response.text[:500]}")
            return False, None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, None

def main():
    print("="*70)
    print("SPORTSDATAIO REPLAY API TEST")
    print("Testing historical data replay capabilities")
    print("="*70)
    
    results = {}
    
    # Test different replay endpoints based on documentation
    print("\n\n📅 TESTING REPLAY DATA ACCESS")
    print("-" * 70)
    
    # Try the example from the documentation
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/scores/json/GamesByDate/2023-NOV-29",
        "Games by Date - Nov 29, 2023 (from replay example)"
    )
    results['replay_games_date'] = success
    
    # Try scores endpoint for same date
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/scores/json/ScoresBasic/2023-NOV-29",
        "Scores Basic - Nov 29, 2023"
    )
    results['replay_scores'] = success
    
    # Try box scores for historical date
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/stats/json/BoxScores/2023-NOV-29",
        "Box Scores - Nov 29, 2023"
    )
    results['replay_boxscores'] = success
    
    # Try team stats for 2023 season
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/scores/json/TeamSeasonStats/2023",
        "Team Season Stats - 2023"
    )
    results['replay_team_stats'] = success
    
    # Try player stats for 2023 season
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/stats/json/PlayerSeasonStats/2023",
        "Player Season Stats - 2023"
    )
    results['replay_player_stats'] = success
    
    # Try historical odds
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/odds/json/GameOddsByDate/2023-NOV-29",
        "Game Odds - Nov 29, 2023"
    )
    results['replay_odds'] = success
    
    # Try a game from that date if we got games
    if results.get('replay_games_date') and data:
        game_id = data[0].get('GameID')
        if game_id:
            success, odds_data = test_replay_endpoint(
                f"{BASE_URL}/v3/cbb/odds/json/GameOddsLineMovement/{game_id}",
                f"Historical Odds Line Movement - Game {game_id}"
            )
            results['replay_line_movement'] = success
    
    # Try tournament bracket
    success, data = test_replay_endpoint(
        f"{BASE_URL}/v3/cbb/scores/json/Tournament/2023",
        "Tournament Bracket - 2023"
    )
    results['replay_tournament'] = success
    
    # SUMMARY
    print("\n\n" + "="*70)
    print("📊 REPLAY API SUMMARY")
    print("="*70)
    
    available = []
    unavailable = []
    
    for feature, is_available in results.items():
        feature_name = feature.replace('replay_', '').replace('_', ' ').title()
        if is_available:
            available.append(feature_name)
            print(f"✅ {feature_name}")
        else:
            unavailable.append(feature_name)
            print(f"❌ {feature_name}")
    
    print("\n" + "="*70)
    print("💡 POTENTIAL USES FOR YOUR PROJECT")
    print("="*70)
    
    if results.get('replay_games_date'):
        print("\n✅ HISTORICAL GAMES DATA:")
        print("   → Backtest your prediction models on past seasons")
        print("   → Train ML models with complete historical datasets")
        print("   → Validate model accuracy against known outcomes")
    
    if results.get('replay_odds'):
        print("\n✅ HISTORICAL ODDS DATA:")
        print("   → Build opening line database from past seasons")
        print("   → Analyze line movement patterns historically")
        print("   → Test betting strategies with real market data")
    
    if results.get('replay_line_movement'):
        print("\n✅ LINE MOVEMENT HISTORY:")
        print("   → Track how lines moved throughout game day")
        print("   → Identify sharp money patterns")
        print("   → Validate line movement prediction algorithms")
    
    if results.get('replay_team_stats'):
        print("\n✅ HISTORICAL TEAM STATS:")
        print("   → Enrich training data with official stats")
        print("   → Compare with KenPom metrics")
        print("   → Build more comprehensive feature sets")
    
    if results.get('replay_player_stats'):
        print("\n✅ HISTORICAL PLAYER STATS:")
        print("   → Add player-level analysis to models")
        print("   → Track individual performance trends")
        print("   → Identify impact of key players")
    
    if results.get('replay_tournament'):
        print("\n✅ TOURNAMENT DATA:")
        print("   → Train March Madness specific models")
        print("   → Analyze seeding vs performance")
        print("   → Build bracket prediction algorithms")
    
    if available:
        print("\n" + "="*70)
        print("🎯 RECOMMENDATION:")
        print("="*70)
        print("The Replay API could be VERY VALUABLE for:")
        print("1. Building comprehensive opening line database from 2023 season")
        print("2. Backtesting your current models against historical data")
        print("3. Training improved models with richer feature sets")
        print("4. Validating betting strategies with real historical odds")
        print("\nThis could solve your opening lines problem AND improve models!")
    else:
        print("\n❌ Replay API not available in free trial")
        print("   → Would require paid subscription to access")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
