import json
import pandas as pd
from predictions import normalize_team_name

# Load ESPN games
df = pd.read_csv('data_files/espn_cbb_current_season.csv')
espn_games = set()
espn_raw = []
for _, row in df.iterrows():
    home = normalize_team_name(row['home_team'])
    away = normalize_team_name(row['away_team'])
    espn_games.add(f'{home} vs {away}')
    espn_raw.append({
        'key': f'{home} vs {away}',
        'home': row['home_team'],
        'away': row['away_team'],
        'normalized_home': home,
        'normalized_away': away
    })

# Load odds
cache = json.load(open('data_files/cache/odds_live.json'))
odds_games = set(cache['odds'].keys())

# Find games that have odds but are NOT in ESPN
odds_only = odds_games - espn_games

print(f'ESPN games: {len(df)}')
print(f'Odds API games: {len(odds_games)}')
print(f'Games with odds but NOT in ESPN: {len(odds_only)}')
print(f'Games in both: {len(espn_games.intersection(odds_games))}')

# Also show ESPN games without odds
espn_only = espn_games - odds_games
print(f'ESPN games without odds: {len(espn_only)}')

if espn_only:
    print('\n=== ESPN GAMES WITHOUT ODDS ===')
    for game_key in sorted(list(espn_only)[:10]):  # Show first 10
        print(f'  {game_key}')
    if len(espn_only) > 10:
        print(f'  ... and {len(espn_only) - 10} more')

if odds_only:
    print('\n=== GAMES WITH ODDS BUT NOT IN ESPN SCHEDULE ===')
    for game_key in sorted(odds_only):
        odds_data = cache['odds'][game_key]
        print(f'\n{game_key}')
        print(f'  Home ML: {odds_data.get("home_moneyline")}')
        print(f'  Away ML: {odds_data.get("away_moneyline")}')
        print(f'  Spread: {odds_data.get("home_spread")} ({odds_data.get("spread_bookmaker")})')
        print(f'  Total: {odds_data.get("total_line")} ({odds_data.get("total_bookmaker")})')
        print(f'  Time: {odds_data.get("commence_time")}')

    print('\n=== CHECKING FOR SIMILAR TEAM NAMES ===')
    print('Looking for potential normalization issues...')

    # Check if any odds-only games have similar names to ESPN games
    for game_key in sorted(odds_only):
        home_team, away_team = game_key.split(' vs ')
        print(f'\n{game_key}:')

        # Check for similar names in ESPN
        similar = []
        for espn_game in espn_raw:
            if (home_team in espn_game['normalized_home'] or
                home_team in espn_game['normalized_away'] or
                away_team in espn_game['normalized_home'] or
                away_team in espn_game['normalized_away']):
                similar.append(f"{espn_game['away']} @ {espn_game['home']}")

        if similar:
            print(f'  Similar ESPN games: {similar}')
        else:
            print('  No similar games in ESPN schedule'
else:
    print('All games with odds are also in ESPN schedule!')