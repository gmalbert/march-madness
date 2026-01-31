import json

data = json.load(open('data_files/upcoming_game_predictions.json'))

print("Games WITH odds:\n")
for g in data:
    gi = g['game_info']
    if gi.get('home_moneyline') is not None:
        print(f"{gi['away_team']} @ {gi['home_team']}")
        print(f"  Home ML: {gi.get('home_moneyline')}")
        print(f"  Away ML: {gi.get('away_moneyline')}")
        print(f"  Home Spread: {gi.get('home_spread')}")
        print(f"  Away Spread: {gi.get('away_spread')}")
        print(f"  Total: {gi.get('total_line')}")
        print()
