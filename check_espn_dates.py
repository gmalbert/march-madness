import pandas as pd
from pathlib import Path

espn_file = Path('data_files/espn_cbb_current_season.csv')
df = pd.read_csv(espn_file)
upcoming = df[df['status'] == 'STATUS_SCHEDULED']

print('ESPN UPCOMING GAMES SAMPLE:')
for i, (_, game) in enumerate(upcoming.head(10).iterrows()):
    print(f'{i+1}. {game["home_team"]} vs {game["away_team"]} | {game["date"]}')

print(f'\nTotal ESPN upcoming: {len(upcoming)}')
print(f'Date range: {upcoming["date"].min()} to {upcoming["date"].max()}')