import json

# Check how many games we have vs betting lines for 2023
with open('data_files/cache/games_2023_regular.json', 'r') as f:
    games_2023 = json.load(f)

with open('data_files/cache/lines_2023_regular.json', 'r') as f:
    lines_2023 = json.load(f)

print('2023 Data:')
print('Games:', len(games_2023))
print('Betting lines:', len(lines_2023))

# Check if game IDs match
games_with_lines = [g for g in lines_2023 if g.get('lines') and len(g['lines']) > 0]
print('Games with betting lines:', len(games_with_lines))

# Check a sample game ID format
if games_2023 and lines_2023:
    print()
    print('Sample game ID formats:')
    print('Games data:', games_2023[0]['homeTeam'], 'vs', games_2023[0]['awayTeam'])
    print('Lines data:', lines_2023[0]['homeTeam'], 'vs', lines_2023[0]['awayTeam'])

    # Check if the team names match for matching
    game_id_games = f'2023_{games_2023[0]["homeTeam"]}_{games_2023[0]["awayTeam"]}'
    game_id_lines = f'2023_{lines_2023[0]["homeTeam"]}_{lines_2023[0]["awayTeam"]}'
    print('Game ID from games:', game_id_games)
    print('Game ID from lines:', game_id_lines)
    print('Match:', game_id_games == game_id_lines)

# Check how many games from the training data have betting lines
import pandas as pd
df = pd.read_csv('data_files/training_data_weighted.csv')
df_2023 = df[df['season'] == 2023]
print()
print('2023 Training Data:')
print('Total games:', len(df_2023))
print('Games with betting spread:', df_2023['betting_spread'].notna().sum())
print('Games with betting over/under:', df_2023['betting_over_under'].notna().sum())
print('Games with home moneyline:', df_2023['home_moneyline'].notna().sum())