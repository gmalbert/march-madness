import pandas as pd

# Read match files (created by scripts/match_teams.py)
kp_path = 'data_files/kenpom_to_espn_matches.csv'
bt_path = 'data_files/bart_to_espn_matches.csv'

kp = pd.read_csv(kp_path)
bt = pd.read_csv(bt_path)

# Missing rows: espn_match is empty string
kp_missing = kp[kp['espn_match'].isna() | (kp['espn_match'].str.strip() == '')].copy()
bt_missing = bt[bt['espn_match'].isna() | (bt['espn_match'].str.strip() == '')].copy()

# If fuzzy suggestions existed earlier they would be in espn_match; include a simple suggestion using difflib as fallback
import difflib

training = pd.read_csv('data_files/training_data_comprehensive.csv')
training_teams = sorted(set(training['home_team'].tolist() + training['away_team'].tolist()))

def suggest(team):
    t = team
    candidates = difflib.get_close_matches(t, training_teams, n=3, cutoff=0.6)
    return ', '.join(candidates)

kp_missing['suggestions'] = kp_missing['kenpom'].apply(suggest)
bt_missing['suggestions'] = bt_missing['bart'].apply(suggest)

kp_missing.to_csv('data_files/kenpom_missing.csv', index=False)
bt_missing.to_csv('data_files/bart_missing.csv', index=False)

print(f"KenPom missing: {len(kp_missing)} -> saved to data_files/kenpom_missing.csv")
print(f"BartTorvik missing: {len(bt_missing)} -> saved to data_files/bart_missing.csv")
