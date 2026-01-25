#!/usr/bin/env python3
"""Test script to verify betting line moneyline matching works with historical data."""

import json
from pathlib import Path

def normalize_team_name(name: str) -> str:
    """Normalize team names for matching."""
    mascots = [
        'Wolverines', 'Hoosiers', 'Cyclones', 'Knights', 'Gators', 'Tigers',
        'Wolfpack', 'Dukes', 'Billikens', 'Bonnies', 'Buckeyes', 'Demon Deacons',
        'Spartans', 'Bears', 'Raiders', 'Razorbacks', 'Commodores', 'Bulldogs',
        'Bruins', 'Boilermakers', 'Buffaloes', 'Jayhawks', 'Wildcats', 'Aggies',
        'Huskies', 'Tar Heels', 'Blue Devils', 'Cardinals', 'Sooners', 'Longhorns',
        'Crimson Tide', 'Volunteers', 'Gamecocks', 'Rebels', 'Broncos', 'Cougars',
        'Panthers', 'Eagles', 'Owls', 'Rams', 'Bulls', 'Golden Knights', 'Mean Green',
        'Golden Gophers', 'Scarlet Knights', 'Friars', 'Yellow Jackets'
    ]
    
    parts = name.split()
    if len(parts) > 1 and parts[-1] in mascots:
        return ' '.join(parts[:-1])
    return name

# Load 2024 betting lines
lines_file = Path('data_files/cache/lines_2024_postseason.json')
with open(lines_file, 'r') as f:
    lines_data = json.load(f)

# Filter to games with actual betting lines
with_lines = [g for g in lines_data if g.get('lines') and len(g['lines']) > 0]

print(f"Total games: {len(lines_data)}")
print(f"Games with betting lines: {len(with_lines)}")
print()

# Test matching for first few games
print("Testing moneyline extraction for first 5 games with lines:")
print("-" * 80)

for i, line in enumerate(with_lines[:5]):
    home_team = line.get('homeTeam')
    away_team = line.get('awayTeam')
    providers = line.get('lines', [])
    
    if providers and len(providers) > 0:
        provider = providers[0]
        home_ml = provider.get('homeMoneyline')
        away_ml = provider.get('awayMoneyline')
        spread = provider.get('spread')
        
        print(f"{i+1}. {away_team} @ {home_team}")
        print(f"   Normalized: {normalize_team_name(away_team)} @ {normalize_team_name(home_team)}")
        print(f"   Home ML: {home_ml}, Away ML: {away_ml}, Spread: {spread}")
        print()

# Test a specific matching scenario
print("\nTesting matching logic:")
print("-" * 80)

# Simulate looking for "Penn State" vs "Michigan"
target_home = "Penn State"
target_away = "Michigan"

found = False
for line in with_lines:
    home_team = line.get('homeTeam')
    away_team = line.get('awayTeam')
    
    # Try various matching strategies
    if ((home_team == target_home and away_team == target_away) or
        (normalize_team_name(home_team) == target_home and normalize_team_name(away_team) == target_away)):
        found = True
        providers = line.get('lines', [])
        if providers:
            provider = providers[0]
            print(f"✅ MATCH FOUND!")
            print(f"   Teams: {away_team} @ {home_team}")
            print(f"   Home ML: {provider.get('homeMoneyline')}")
            print(f"   Away ML: {provider.get('awayMoneyline')}")
            print(f"   Spread: {provider.get('spread')}")
            print(f"   O/U: {provider.get('overUnder')}")
        break

if not found:
    print(f"❌ No match found for {target_away} @ {target_home}")
    print("\nAvailable games:")
    for line in with_lines[:10]:
        print(f"   {line.get('awayTeam')} @ {line.get('homeTeam')}")
