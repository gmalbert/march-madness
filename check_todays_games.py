import json
from pathlib import Path
from datetime import datetime

p = Path('data_files/upcoming_game_predictions.json')
data = json.load(open(p))

print(f'Total games in file: {len(data)}')

today = datetime.now().strftime('%Y-%m-%d')
print(f'Today is: {today}')

# Check first few game dates
print('\nFirst 5 games with dates:')
for i, g in enumerate(data[:5]):
    game_date = g.get('game_info', {}).get('date', 'NO DATE')
    home = g.get('game_info', {}).get('home_team', 'Unknown')
    away = g.get('game_info', {}).get('away_team', 'Unknown')
    print(f'  {i+1}. {away} @ {home} - {game_date}')

# Filter today's games
todays_games = [g for g in data if g.get('game_info', {}).get('date', '').startswith(today)]
print(f'\nGames scheduled for today ({today}): {len(todays_games)}')

for g in todays_games:
    home = g.get('game_info', {}).get('home_team', 'Unknown')
    away = g.get('game_info', {}).get('away_team', 'Unknown')
    date = g.get('game_info', {}).get('date', 'NO DATE')
    print(f'  {away} @ {home} - {date}')
