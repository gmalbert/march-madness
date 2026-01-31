import pandas as pd
from pathlib import Path
from fetch_live_odds import fetch_live_odds
from predictions import normalize_team_name

# Get ESPN games
espn_file = Path('data_files/espn_cbb_current_season.csv')
df = pd.read_csv(espn_file)
upcoming = df[df['status'] == 'STATUS_SCHEDULED']

# Get odds games
odds = fetch_live_odds()

print('MATCHING ESPN GAMES TO ODDS API:')
matches = 0
for _, game in upcoming.iterrows():
    home_norm = normalize_team_name(game['home_team'])
    away_norm = normalize_team_name(game['away_team'])
    key = f'{home_norm} vs {away_norm}'
    
    if key in odds:
        matches += 1
        print(f'✓ {game["home_team"]} vs {game["away_team"]}')
        print(f'  Key: {key}')
        ml = odds[key].get('home_moneyline', 'N/A')
        print(f'  Moneyline: {ml}')
        print()

print(f'Total matches: {matches}/{len(upcoming)} ESPN games')
print(f'Odds API has: {len(odds)} games')