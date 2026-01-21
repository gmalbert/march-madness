import pandas as pd
from data_collection import fetch_adjusted_efficiency, fetch_team_stats
from predictions import normalize_team_name

# Load ESPN games
df = pd.read_csv('data_files/espn_cbb_current_season.csv')
espn_teams = set(df.home_team.tolist() + df.away_team.tolist())

# Get CBBD data
eff_data = fetch_adjusted_efficiency(2025)  # Using 2025 data as fallback
stats_data = fetch_team_stats(2025)

# Extract team names
eff_teams = set(e.get('team', '') for e in eff_data)
stats_teams = set(s.get('team', '') for s in stats_data)

print('ESPN upcoming games:')
for idx, row in df.iterrows():
    print(f'  {row["away_team"]} @ {row["home_team"]}')

print(f'\nTotal ESPN teams: {len(espn_teams)}')
print(f'CBBD efficiency teams: {len(eff_teams)}')
print(f'CBBD stats teams: {len(stats_teams)}')

print('\nTeam data availability check:')
missing_teams = []
for espn_team in sorted(espn_teams):
    normalized = normalize_team_name(espn_team)
    has_eff = normalized in eff_teams
    has_stats = normalized in stats_teams
    status = 'OK' if (has_eff and has_stats) else 'MISSING DATA'
    if status == 'MISSING DATA':
        missing_teams.append((espn_team, normalized))
    print(f'  {espn_team} -> {normalized}: EFF={has_eff}, STATS={has_stats} [{status}]')

print(f'\nTeams missing CBBD data: {len(missing_teams)}')
for espn_name, normalized_name in missing_teams:
    print(f'  {espn_name} -> {normalized_name}')