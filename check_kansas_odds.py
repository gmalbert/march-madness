import json

data = json.load(open('data_files/upcoming_game_predictions.json'))
kansas_games = [g for g in data if 'Kansas' in g['game_info']['home_team'] or 'Kansas' in g['game_info']['away_team']]
for game in kansas_games:
    info = game['game_info']
    has_odds = info.get('home_moneyline') is not None
    print(f"Kansas game: {info['away_team']} @ {info['home_team']}")
    print(f"  Has odds: {has_odds}")
    if has_odds:
        print(f"  Home ML: {info.get('home_moneyline')}, Away ML: {info.get('away_moneyline')}")
    print()