import json
from pathlib import Path
from datetime import datetime, timezone, timedelta

p = Path('data_files/upcoming_game_predictions.json')
data = json.load(open(p))

print(f'Total games in file: {len(data)}')

now_local = datetime.now()
print(f'Current local time: {now_local}')
print(f'Current local date: {now_local.date()}')

# Check first few game dates with timezone conversion
print('\nFirst 5 games with UTC and local times:')
for i, g in enumerate(data[:5]):
    game_date_str = g.get('game_info', {}).get('date', 'NO DATE')
    home = g.get('game_info', {}).get('home_team', 'Unknown')
    away = g.get('game_info', {}).get('away_team', 'Unknown')
    
    if game_date_str != 'NO DATE':
        try:
            # Parse UTC
            game_date_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
            # Convert to local
            game_date_local = game_date_utc.replace(tzinfo=timezone.utc).astimezone(tz=None)
            
            print(f'\n{i+1}. {away} @ {home}')
            print(f'   UTC: {game_date_utc}')
            print(f'   Local: {game_date_local}')
            print(f'   Local date: {game_date_local.date()}')
        except Exception as e:
            print(f'\n{i+1}. {away} @ {home} - ERROR: {e}')
    else:
        print(f'\n{i+1}. {away} @ {home} - NO DATE')

# Filter for today's games (local time)
three_days_from_now = now_local + timedelta(days=3)
todays_games = []

for g in data:
    game_date_str = g.get('game_info', {}).get('date', '')
    if game_date_str:
        try:
            game_date_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
            game_date_local = game_date_utc.replace(tzinfo=timezone.utc).astimezone(tz=None)
            
            if now_local.date() <= game_date_local.date() <= three_days_from_now.date():
                todays_games.append(g)
        except:
            pass

print(f'\n\nGames in next 3 days (local time): {len(todays_games)}')

# Show games by local date
from collections import defaultdict
games_by_date = defaultdict(list)
for g in todays_games:
    game_date_str = g.get('game_info', {}).get('date', '')
    try:
        game_date_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
        game_date_local = game_date_utc.replace(tzinfo=timezone.utc).astimezone(tz=None)
        date_key = game_date_local.strftime('%A, %B %d')
    except:
        date_key = 'Unknown'
    games_by_date[date_key].append(g)

print('\nGames by local date:')
for date_key in sorted(games_by_date.keys()):
    print(f'  {date_key}: {len(games_by_date[date_key])} games')
