import pandas as pd

df = pd.read_csv('data_files/espn_cbb_current_season.csv')
print(f'ESPN games loaded: {len(df)}')

print('\nFirst 10 games:')
for i, row in df.head(10).iterrows():
    away = row['away_team']
    home = row['home_team']
    date = row.get('date', 'No date')
    print(f'{i+1}. {away} @ {home} - {date}')

print('\nSearching for Kansas games:')
kansas_games = df[df['home_team'].str.contains('Kansas', case=False) | df['away_team'].str.contains('Kansas', case=False)]
if len(kansas_games) > 0:
    for i, row in kansas_games.iterrows():
        away = row['away_team']
        home = row['home_team']
        date = row.get('date', 'No date')
        print(f'  {away} @ {home} - {date}')
else:
    print('  No Kansas games found in ESPN data')

print('\nAll teams in ESPN data (first 20):')
all_teams = set()
for i, row in df.iterrows():
    all_teams.add(row['home_team'])
    all_teams.add(row['away_team'])

for team in sorted(list(all_teams))[:20]:
    print(f'  {team}')
