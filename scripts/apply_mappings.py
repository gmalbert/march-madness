import pandas as pd

# Read the user-updated mappings
kp_updated = pd.read_csv('data_files/kenpom_missing-updated.csv', header=1)
bt_updated = pd.read_csv('data_files/bart_missing-updated.csv', header=1)

# Filter to only rows with valid team names and valid mappings
kp_updated = kp_updated[kp_updated['kenpom'].notna() & (kp_updated['kenpom'].astype(str).str.strip() != '')]
bt_updated = bt_updated[bt_updated['kenpom'].notna() & (bt_updated['kenpom'].astype(str).str.strip() != '')]

# Create mapping dictionaries from user-confirmed entries (excluding empty espn_match)
kp_mappings = {}
for _, row in kp_updated.iterrows():
    kp_team = row['kenpom']
    esp_match = row['espn_match']
    if pd.notna(esp_match) and str(esp_match).strip() != '':
        kp_mappings[kp_team] = str(esp_match).strip()

bt_mappings = {}
for _, row in bt_updated.iterrows():
    bt_team = row['kenpom']  # column name is 'kenpom' in both files
    esp_match = row['espn_match']
    if pd.notna(esp_match) and str(esp_match).strip() != '':
        bt_mappings[bt_team] = str(esp_match).strip()

print(f"KenPom confirmed mappings: {len(kp_mappings)}")
print(f"BartTorvik confirmed mappings: {len(bt_mappings)}")

# Read original match files
kp_matches = pd.read_csv('data_files/kenpom_to_espn_matches.csv')
bt_matches = pd.read_csv('data_files/bart_to_espn_matches.csv')

# Apply user mappings
for idx, row in kp_matches.iterrows():
    team = row['kenpom']
    if team in kp_mappings:
        kp_matches.at[idx, 'espn_match'] = kp_mappings[team]

for idx, row in bt_matches.iterrows():
    team = row['bart']
    if team in bt_mappings:
        bt_matches.at[idx, 'espn_match'] = bt_mappings[team]

# Save updated match files
kp_matches.to_csv('data_files/kenpom_to_espn_matches.csv', index=False)
bt_matches.to_csv('data_files/bart_to_espn_matches.csv', index=False)

# Count final stats
kp_mapped = kp_matches[kp_matches['espn_match'].notna() & (kp_matches['espn_match'].astype(str).str.strip() != '')].shape[0]
kp_unmapped = kp_matches[kp_matches['espn_match'].isna() | (kp_matches['espn_match'].astype(str).str.strip() == '')].shape[0]

bt_mapped = bt_matches[bt_matches['espn_match'].notna() & (bt_matches['espn_match'].astype(str).str.strip() != '')].shape[0]
bt_unmapped = bt_matches[bt_matches['espn_match'].isna() | (bt_matches['espn_match'].astype(str).str.strip() == '')].shape[0]

print(f"\nKenPom final: {kp_mapped} mapped, {kp_unmapped} unmapped (total {len(kp_matches)})")
print(f"BartTorvik final: {bt_mapped} mapped, {bt_unmapped} unmapped (total {len(bt_matches)})")

# Show unmapped teams
kp_unmapped_teams = kp_matches[kp_matches['espn_match'].isna() | (kp_matches['espn_match'].astype(str).str.strip() == '')]['kenpom'].tolist()
bt_unmapped_teams = bt_matches[bt_matches['espn_match'].isna() | (bt_matches['espn_match'].astype(str).str.strip() == '')]['bart'].tolist()

print(f"\nKenPom unmapped teams: {kp_unmapped_teams}")
print(f"BartTorvik unmapped teams: {bt_unmapped_teams}")
