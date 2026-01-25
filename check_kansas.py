import pandas as pd
from predictions import normalize_team_name
import json

# Load ESPN games
df = pd.read_csv('data_files/espn_cbb_current_season.csv')
espn_games = set()
for _, row in df.iterrows():
    home = normalize_team_name(row['home_team'])
    away = normalize_team_name(row['away_team'])
    espn_games.add(f'{home} vs {away}')
    if 'Kansas' in row['home_team'] or 'Kansas' in row['away_team']:
        print(f'ESPN: {row["away_team"]} -> {away}')
        print(f'ESPN: {row["home_team"]} -> {home}')
        print(f'ESPN key: {away} vs {home}')

# Load odds
cache = json.load(open('data_files/cache/odds_live.json'))
odds_games = set(cache['odds'].keys())

# Check Kansas games in odds
for game_key in odds_games:
    if 'Kansas' in game_key:
        print(f'Odds key: {game_key}')

# Check if they match
kansas_espn = [k for k in espn_games if 'Kansas' in k]
kansas_odds = [k for k in odds_games if 'Kansas' in k]
print(f'\nESPN Kansas games: {kansas_espn}')
print(f'Odds Kansas games: {kansas_odds}')

# Check intersection
both = set(kansas_espn).intersection(set(kansas_odds))
print(f'Matching Kansas games: {both}')
