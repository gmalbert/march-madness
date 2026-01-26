import json

data = json.load(open('data_files/upcoming_game_predictions.json'))

# Find Washington games
wash_games = [g for g in data if 'Washington' in g['game_info'].get('home_team', '') or 'Washington' in g['game_info'].get('away_team', '')]

print(f'Found {len(wash_games)} Washington games\n')

for g in wash_games:
    print(f"Game: {g['game_info']['away_team']} @ {g['game_info']['home_team']}")
    print(f"  Home ML: {g['game_info'].get('home_moneyline')}")
    print(f"  Away ML: {g['game_info'].get('away_moneyline')}")
    print(f"  Home Spread: {g['game_info'].get('home_spread')}")
    print(f"  Away Spread: {g['game_info'].get('away_spread')}")
    print(f"  Total: {g['game_info'].get('total_line')}")
    print()
