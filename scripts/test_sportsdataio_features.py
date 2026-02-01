#!/usr/bin/env python3
"""
Test SportsDataIO Free Trial for Non-Betting Data

Check what useful data is available in the free trial that could enhance
our current site and modeling capabilities.
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
BASE_URL = "https://api.sportsdata.io/v3/cbb"
session = requests.Session()
session.headers.update({"Ocp-Apim-Subscription-Key": SPORTS_DATA_IO_KEY})

def test_endpoint(endpoint_url, description):
    """Test an endpoint and return status and sample data."""
    print(f"\n{'='*70}")
    print(f"Testing: {description}")
    print(f"URL: {endpoint_url}")
    try:
        response = session.get(endpoint_url)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            if isinstance(data, list):
                print(f"✅ SUCCESS - Retrieved {len(data)} items")
                if data:
                    print(f"\nSample item (first in list):")
                    print(json.dumps(data[0], indent=2, default=str)[:1000] + "...")
                return True, data[:3]  # Return first 3 items
            elif isinstance(data, dict):
                print(f"✅ SUCCESS - Retrieved data object")
                print(f"\nSample data:")
                print(json.dumps(data, indent=2, default=str)[:1000] + "...")
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
        else:
            print(f"❌ FAILED - Status {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False, None

def main():
    print("="*70)
    print("SPORTSDATAIO FREE TRIAL - FEATURE DISCOVERY")
    print("Testing what data is available beyond betting odds")
    print("="*70)
    
    results = {}
    
    # 1. TEAM DATA
    print("\n\n🏀 TEAM DATA")
    print("-" * 70)
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/teams",
        "All Teams (basic info)"
    )
    results['teams_basic'] = success
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/TeamSeasonStats/2025",
        "Team Season Stats (2025)"
    )
    results['team_season_stats'] = success
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/Standings/2025",
        "Conference Standings (2025)"
    )
    results['standings'] = success
    
    # 2. PLAYER DATA
    print("\n\n👤 PLAYER DATA")
    print("-" * 70)
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/Players",
        "All Active Players"
    )
    results['players'] = success
    
    success, data = test_endpoint(
        f"{BASE_URL}/stats/json/PlayerSeasonStats/2025",
        "Player Season Stats (2025)"
    )
    results['player_season_stats'] = success
    
    # 3. SCHEDULE & GAME DATA
    print("\n\n📅 SCHEDULE & GAME DATA")
    print("-" * 70)
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/GamesByDate/2025-JAN-25",
        "Games by Date (yesterday)"
    )
    results['games_by_date'] = success
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/AreAnyGamesInProgress",
        "Are Games In Progress"
    )
    results['games_in_progress'] = success
    
    # 4. TOURNAMENT DATA
    print("\n\n🏆 TOURNAMENT DATA")
    print("-" * 70)
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/Tournament/2024",
        "Tournament Bracket (2024)"
    )
    results['tournament'] = success
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/LeagueHierarchy",
        "League Hierarchy (Conferences)"
    )
    results['league_hierarchy'] = success
    
    # 5. INJURY DATA
    print("\n\n🏥 INJURY DATA")
    print("-" * 70)
    
    success, data = test_endpoint(
        f"{BASE_URL}/scores/json/InjuredPlayers",
        "Injured Players"
    )
    results['injuries'] = success
    
    # 6. LIVE GAME DATA
    print("\n\n📊 LIVE GAME DATA")
    print("-" * 70)
    
    # Try to get a recent game ID first
    try:
        response = session.get(f"{BASE_URL}/scores/json/GamesByDate/2025-JAN-25")
        if response.status_code == 200:
            games = response.json()
            if games:
                game_id = games[0]['GameID']
                success, data = test_endpoint(
                    f"{BASE_URL}/stats/json/BoxScore/{game_id}",
                    f"Box Score (Game ID: {game_id})"
                )
                results['box_score'] = success
    except:
        results['box_score'] = False
    
    # FINAL SUMMARY
    print("\n\n" + "="*70)
    print("📊 SUMMARY - AVAILABLE FEATURES IN FREE TRIAL")
    print("="*70)
    
    available_features = []
    unavailable_features = []
    
    for feature, available in results.items():
        if available:
            available_features.append(feature)
            print(f"✅ {feature.replace('_', ' ').title()}")
        else:
            unavailable_features.append(feature)
            print(f"❌ {feature.replace('_', ' ').title()}")
    
    print("\n" + "="*70)
    print("💡 RECOMMENDATIONS FOR YOUR SITE/MODELING")
    print("="*70)
    
    if results.get('team_season_stats'):
        print("✅ Team Stats: Can enhance models with official stats beyond KenPom")
        print("   → Add team shooting %, rebounding, turnovers, etc.")
        
    if results.get('player_season_stats'):
        print("✅ Player Stats: Add player-level analysis to your models")
        print("   → Identify key players, track performance trends")
        
    if results.get('injuries'):
        print("✅ Injuries: Real-time injury data for more accurate predictions")
        print("   → Adjust predictions when key players are out")
        
    if results.get('tournament'):
        print("✅ Tournament: Official bracket and seeding data")
        print("   → Enhanced March Madness predictions with seeding info")
        
    if results.get('league_hierarchy'):
        print("✅ Standings: Conference standings and rankings")
        print("   → Display current standings on your site")
        
    if results.get('games_by_date'):
        print("✅ Live Scores: Real-time game scores and status")
        print("   → Replace or supplement ESPN data source")
        
    if not available_features:
        print("❌ No useful data available in free trial")
        print("   → Free trial appears limited to basic team info only")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
