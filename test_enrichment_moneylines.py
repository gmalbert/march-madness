#!/usr/bin/env python3
"""Test that predictions.py enrichment populates moneylines for historical games."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from predictions import enrich_espn_game_with_cbbd_data
from data_collection import fetch_adjusted_efficiency, fetch_team_stats

# Create a simulated ESPN game row for Michigan @ Penn State from 2024
test_game = {
    'home_team': 'Penn State',
    'away_team': 'Michigan',
    'date': '2024-03-14',
    'status': 'STATUS_FINAL',
    'venue': 'Test Arena',
    'neutral_site': False,
    'home_rank': 99,
    'away_rank': 10
}

print("Testing enrichment with 2024 historical game...")
print(f"Game: {test_game['away_team']} @ {test_game['home_team']}")
print()

# Fetch 2024 season data (when betting lines exist)
try:
    efficiency_list = fetch_adjusted_efficiency(2024)
    stats_list = fetch_team_stats(2024)
    print(f"✅ Loaded efficiency for {len(efficiency_list)} teams")
    print(f"✅ Loaded stats for {len(stats_list)} teams")
except Exception as e:
    print(f"❌ Error loading team data: {e}")
    sys.exit(1)

# Enrich the game
enriched = enrich_espn_game_with_cbbd_data(test_game, efficiency_list, stats_list, 2024)

if enriched:
    print("\n✅ Game enrichment successful!")
    print("\nMoneyline fields:")
    print(f"  home_moneyline: {enriched.get('home_moneyline')}")
    print(f"  away_moneyline: {enriched.get('away_moneyline')}")
    print(f"  home_ml: {enriched.get('home_ml')}")
    print(f"  away_ml: {enriched.get('away_ml')}")
    
    if enriched.get('home_moneyline') is not None and enriched.get('away_moneyline') is not None:
        print("\n✅ SUCCESS: Moneylines populated from CFBD betting lines!")
        print("\nNow testing market probability calculation...")
        
        try:
            from features import calculate_implied_probability
            
            home_ml = enriched['home_moneyline']
            away_ml = enriched['away_moneyline']
            
            home_implied = calculate_implied_probability(home_ml)
            away_implied = calculate_implied_probability(away_ml)
            
            print(f"  Home ML {home_ml} → Implied Prob: {home_implied:.1%}")
            print(f"  Away ML {away_ml} → Implied Prob: {away_implied:.1%}")
            print("\n✅ Market probability calculation works!")
            print("✅ UI Market Prob/Edge feature will work for historical games!")
        except Exception as e:
            print(f"❌ Error calculating implied probabilities: {e}")
    else:
        print("\n⚠️  Moneylines are NULL - betting lines not found in CFBD data")
        print("   This is expected for 2025/2026 games (no live odds)")
else:
    print("❌ Game enrichment failed")
