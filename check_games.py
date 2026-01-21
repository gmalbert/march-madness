import pandas as pd
from datetime import datetime
import pytz

print('Current date:', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

df = pd.read_csv('data_files/espn_cbb_current_season.csv')
print(f'Total games: {len(df)}')

df['date_dt'] = pd.to_datetime(df['date'])
# Convert to UTC for comparison
now_utc = pd.Timestamp.now(tz='UTC')
upcoming = df[df['date_dt'] > now_utc]
past = df[df['date_dt'] <= now_utc]

print(f'Upcoming games: {len(upcoming)}')
print(f'Past games: {len(past)}')

if len(upcoming) > 0:
    print('Upcoming games:')
    for idx, row in upcoming.iterrows():
        print(f'  {row["away_team"]} @ {row["home_team"]}: {row["date_dt"].strftime("%Y-%m-%d %H:%M")}')
else:
    print('No upcoming games found')

print('\nAll games:')
for idx, row in df.iterrows():
    status = 'FUTURE' if row['date_dt'] > now_utc else 'PAST'
    print(f'  {row["away_team"]} @ {row["home_team"]}: {row["date_dt"].strftime("%Y-%m-%d %H:%M")} [{status}]')