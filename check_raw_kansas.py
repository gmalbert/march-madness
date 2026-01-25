import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds'

params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'h2h',
    'oddsFormat': 'american'
}

response = requests.get(BASE_URL, params=params, timeout=30)
data = response.json()

# Find Kansas games
for game in data:
    if 'Kansas' in game.get('home_team', '') or 'Kansas' in game.get('away_team', ''):
        print('Raw API data for Kansas game:')
        print(f'  home_team: {game.get("home_team")}')
        print(f'  away_team: {game.get("away_team")}')
        print(f'  commence_time: {game.get("commence_time")}')
        break