import json

data = json.load(open('data_files/upcoming_game_predictions.json'))

games_with_ml = 0
games_with_spread = 0
games_with_total = 0
games_with_no_odds = 0

for g in data:
    gi = g['game_info']
    has_ml = gi.get('home_moneyline') is not None
    has_spread = gi.get('home_spread') is not None
    has_total = gi.get('total_line') is not None
    
    if has_ml:
        games_with_ml += 1
    if has_spread:
        games_with_spread += 1
    if has_total:
        games_with_total += 1
    
    if not has_ml and not has_spread and not has_total:
        games_with_no_odds += 1
        print(f"No odds: {gi['away_team']} @ {gi['home_team']}")

print(f"\n\nSummary:")
print(f"Total games: {len(data)}")
print(f"Games with moneyline: {games_with_ml}")
print(f"Games with spread: {games_with_spread}")
print(f"Games with total: {games_with_total}")
print(f"Games with NO odds at all: {games_with_no_odds}")
