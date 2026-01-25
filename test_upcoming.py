import requests
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds'

# Test 1: No date filters (all upcoming games with markets)
params_no_filter = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'h2h',
    'oddsFormat': 'american'
}

print("Fetching with NO date filters (all upcoming)...")
response = requests.get(BASE_URL, params=params_no_filter, timeout=30)
data_no_filter = response.json()

print(f'\nTotal games (no filter): {len(data_no_filter)}')
print('\nGames:')
for game in data_no_filter:
    print(f'{game["away_team"]} @ {game["home_team"]} - {game["commence_time"]}')
