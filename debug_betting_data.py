"""
Debug script to analyze betting data in predictions
"""
import json
from pathlib import Path

DATA_DIR = Path("data_files")

# Load the predictions file
pred_file = DATA_DIR / "upcoming_game_predictions.json"

if not pred_file.exists():
    print("❌ No upcoming_game_predictions.json file found!")
    exit(1)

with open(pred_file, 'r') as f:
    games = json.load(f)

print(f"📊 Analyzing {len(games)} games in upcoming_game_predictions.json\n")

# Statistics
total_games = len(games)
has_spread = 0
has_total = 0
has_moneyline = 0
has_all_three = 0

# Track field names we find
spread_fields = set()
total_fields = set()
ml_fields = set()

# Analyze each game
for game in games:
    game_info = game.get('game_info', {})
    
    # Check for spread
    if game_info.get('home_spread') is not None or game_info.get('away_spread') is not None:
        has_spread += 1
        if game_info.get('home_spread') is not None:
            spread_fields.add('home_spread')
        if game_info.get('away_spread') is not None:
            spread_fields.add('away_spread')
    
    # Check for total
    if game_info.get('total_line') is not None:
        has_total += 1
        total_fields.add('total_line')
    
    # Check for moneyline
    if game_info.get('home_moneyline') is not None or game_info.get('away_moneyline') is not None:
        has_moneyline += 1
        if game_info.get('home_moneyline') is not None:
            ml_fields.add('home_moneyline')
        if game_info.get('away_moneyline') is not None:
            ml_fields.add('away_moneyline')
    
    # Check for all three
    if (game_info.get('home_spread') is not None and 
        game_info.get('total_line') is not None and
        game_info.get('home_moneyline') is not None):
        has_all_three += 1

print("=" * 60)
print("BETTING DATA STATISTICS")
print("=" * 60)
print(f"Total Games:              {total_games}")
print(f"Games with Spread:        {has_spread} ({has_spread/total_games*100:.1f}%)")
print(f"Games with Total:         {has_total} ({has_total/total_games*100:.1f}%)")
print(f"Games with Moneyline:     {has_moneyline} ({has_moneyline/total_games*100:.1f}%)")
print(f"Games with ALL betting:   {has_all_three} ({has_all_three/total_games*100:.1f}%)")
print()

print("=" * 60)
print("FIELD NAMES FOUND")
print("=" * 60)
print(f"Spread fields:    {spread_fields}")
print(f"Total fields:     {total_fields}")
print(f"Moneyline fields: {ml_fields}")
print()

# Sample some games with betting data
print("=" * 60)
print("SAMPLE GAMES WITH BETTING DATA")
print("=" * 60)
games_with_betting = [g for g in games if g.get('game_info', {}).get('home_spread') is not None]
for i, game in enumerate(games_with_betting[:5]):
    info = game['game_info']
    print(f"\nGame {i+1}: {info['away_team']} @ {info['home_team']}")
    print(f"  Spread: {info.get('home_spread')}")
    print(f"  Total:  {info.get('total_line')}")
    print(f"  ML:     {info.get('home_moneyline')} / {info.get('away_moneyline')}")

# Sample games WITHOUT betting data
print("\n" + "=" * 60)
print("SAMPLE GAMES WITHOUT BETTING DATA")
print("=" * 60)
games_without_betting = [g for g in games if g.get('game_info', {}).get('home_spread') is None]
for i, game in enumerate(games_without_betting[:5]):
    info = game['game_info']
    print(f"\nGame {i+1}: {info['away_team']} @ {info['home_team']}")
    print(f"  Date: {info.get('date')}")
    print(f"  All fields: {list(info.keys())}")

print("\n" + "=" * 60)
print("DIAGNOSIS")
print("=" * 60)

if has_all_three > 0:
    print(f"✅ {has_all_three} games have complete betting data")
    print("   The pages should be working for these games.")
else:
    print("❌ NO games have complete betting data")
    print("   This means fetch_live_odds.py or generate_predictions.py didn't work correctly")

if has_spread < total_games * 0.5:
    print(f"\n⚠️  Less than 50% of games have spread data")
    print("   Possible causes:")
    print("   1. Odds API doesn't have lines for these games yet")
    print("   2. Team name matching issues between ESPN and Odds API")
    print("   3. fetch_live_odds.py didn't run successfully")
