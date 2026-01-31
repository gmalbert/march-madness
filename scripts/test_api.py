import requests
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = 'https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds'

now_utc = datetime.now(timezone.utc)
commence_time_from = now_utc.strftime('%Y-%m-%dT%H:%M:%SZ')
commence_time_to = (now_utc + timedelta(days=7)).strftime('%Y-%m-%dT%H:%M:%SZ')

params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'h2h',
    'oddsFormat': 'american',
    'commenceTimeFrom': commence_time_from,
    'commenceTimeTo': commence_time_to
}

response = requests.get(BASE_URL, params=params, timeout=30)
data = response.json()

print(f'Total games: {len(data)}')
print('\nRaw team names from API:')
for game in data:
    print(f'{game["away_team"]} @ {game["home_team"]} - {game["commence_time"]}')
