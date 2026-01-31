import pandas as pd
from pathlib import Path
from fetch_live_odds import fetch_live_odds
from predictions import normalize_team_name

# Load data
espn_file = Path('data_files/espn_cbb_current_season.csv')
df = pd.read_csv(espn_file)
upcoming = df[df['status'] == 'STATUS_SCHEDULED']

odds = fetch_live_odds()

print('Games with live odds:')
for _, game in upcoming.iterrows():
    home_norm = normalize_team_name(game['home_team'])
    away_norm = normalize_team_name(game['away_team'])
    key = f'{home_norm} vs {away_norm}'
    if key in odds:
        print(f'  {game["home_team"]} vs {game["away_team"]}')
        print(f'    Key: {key}')
        print(f'    ML: {odds[key].get("home_moneyline")} / {odds[key].get("away_moneyline")}')