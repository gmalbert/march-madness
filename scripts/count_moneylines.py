import json
data = json.load(open('data_files/upcoming_game_predictions.json'))
count = 0
for g in data:
    if g['game_info'].get('home_moneyline'):
        count += 1
        if count <= 5:  # Show first 5
            print(f"{g['game_info']['home_team']} vs {g['game_info']['away_team']}: {g['game_info']['home_moneyline']} / {g['game_info']['away_moneyline']}")
print(f"Total games with moneyline: {count}/50 ({count/50*100:.1f}%)")