import json

data = json.load(open('data_files/upcoming_game_predictions.json'))
with_odds = sum(1 for g in data if g['game_info'].get('home_moneyline'))
print(f'Games with odds: {with_odds}/{len(data)}')

games_with_odds = [g for g in data if g['game_info'].get('home_moneyline')]
print('\nGames with odds:')
for g in games_with_odds:
    away = g['game_info']['away_team']
    home = g['game_info']['home_team']
    home_ml = g['game_info']['home_moneyline']
    away_ml = g['game_info']['away_moneyline']
    print(f"  {away} @ {home} (Home: {home_ml}, Away: {away_ml})")
