import pandas as pd
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
from predictions import normalize_team_name

# Load environment
load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

# Get ESPN games
espn_file = Path('data_files/espn_cbb_current_season.csv')
df = pd.read_csv(espn_file)
upcoming = df[df['status'] == 'STATUS_SCHEDULED']

print("=== ESPN GAMES ANALYSIS ===")
espn_games = []
for _, game in upcoming.iterrows():
    home_raw = game['home_team']
    away_raw = game['away_team']
    home_norm = normalize_team_name(home_raw)
    away_norm = normalize_team_name(away_raw)
    espn_games.append({
        'home_raw': home_raw,
        'away_raw': away_raw,
        'home_norm': home_norm,
        'away_norm': away_norm,
        'key': f"{home_norm} vs {away_norm}"
    })

print(f"ESPN has {len(espn_games)} games")
print("Sample ESPN games:")
for i, game in enumerate(espn_games[:5]):
    print(f"  {i+1}. {game['home_raw']} vs {game['away_raw']}")
    print(f"     -> {game['key']}")

# Get Odds API games
BASE_URL = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds'
params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'h2h,spreads,totals',
    'oddsFormat': 'american'
}

response = requests.get(BASE_URL, params=params, timeout=30)
odds_data = response.json()

print(f"\n=== ODDS API GAMES ANALYSIS ===")
print(f"API returned {len(odds_data)} games")
odds_games = []
for game in odds_data:
    home_raw = game.get('home_team', 'Unknown')
    away_raw = game.get('away_team', 'Unknown')
    home_norm = normalize_team_name(home_raw)
    away_norm = normalize_team_name(away_raw)
    odds_games.append({
        'home_raw': home_raw,
        'away_raw': away_raw,
        'home_norm': home_norm,
        'away_norm': away_norm,
        'key': f"{home_norm} vs {away_norm}"
    })

print("Sample Odds API games:")
for i, game in enumerate(odds_games[:5]):
    print(f"  {i+1}. {game['home_raw']} vs {game['away_raw']}")
    print(f"     -> {game['key']}")

# Find matches
print(f"\n=== MATCHING ANALYSIS ===")
espn_keys = {game['key'] for game in espn_games}
odds_keys = {game['key'] for game in odds_games}
matches = espn_keys & odds_keys

print(f"ESPN unique keys: {len(espn_keys)}")
print(f"Odds API unique keys: {len(odds_keys)}")
print(f"Matching keys: {len(matches)}")

if matches:
    print("Matching games:")
    for key in sorted(list(matches)[:10]):  # First 10 matches
        espn_game = next(g for g in espn_games if g['key'] == key)
        odds_game = next(g for g in odds_games if g['key'] == key)
        print(f"  {key}")
        print(f"    ESPN: {espn_game['home_raw']} vs {espn_game['away_raw']}")
        print(f"    Odds: {odds_game['home_raw']} vs {odds_game['away_raw']}")

# Check for near misses
print(f"\n=== NEAR MISS ANALYSIS ===")
print("ESPN keys not in Odds API:")
missing_from_odds = espn_keys - odds_keys
for key in sorted(list(missing_from_odds)[:10]):
    espn_game = next(g for g in espn_games if g['key'] == key)
    print(f"  {key} (ESPN: {espn_game['home_raw']} vs {espn_game['away_raw']})")

print(f"\nOdds API keys not in ESPN:")
missing_from_espn = odds_keys - espn_keys
for key in sorted(list(missing_from_espn)[:10]):
    odds_game = next(g for g in odds_games if g['key'] == key)
    print(f"  {key} (Odds: {odds_game['home_raw']} vs {odds_game['away_raw']})")