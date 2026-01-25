import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds'
params = {'apiKey': API_KEY, 'regions': 'us', 'markets': 'h2h,spreads,totals', 'oddsFormat': 'american'}

response = requests.get(BASE_URL, params=params, timeout=30)
odds_data = response.json()

from predictions import normalize_team_name

missing_mascots = set()
for game in odds_data:
    home_raw = game.get('home_team', '')
    away_raw = game.get('away_team', '')

    home_norm = normalize_team_name(home_raw)
    away_norm = normalize_team_name(away_raw)

    if home_norm == home_raw and ' ' in home_raw:
        parts = home_raw.split()
        if len(parts) > 1:
            mascot = ' '.join(parts[1:])
            missing_mascots.add(mascot)

    if away_norm == away_raw and ' ' in away_raw:
        parts = away_raw.split()
        if len(parts) > 1:
            mascot = ' '.join(parts[1:])
            missing_mascots.add(mascot)

print('Mascots not recognized by normalization:')
for mascot in sorted(missing_mascots):
    print(f'  "{mascot}",')