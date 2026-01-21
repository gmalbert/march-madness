import sys
sys.path.append('.')
from predictions import get_upcoming_games

games = get_upcoming_games()
print(f'Loaded {len(games)} games:')
for game in games:
    print(f'  {game["away_team"]} @ {game["home_team"]}')