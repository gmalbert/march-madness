import pandas as pd
import difflib
import re

# Replicate normalize_team_name logic for canonicalization
mascots = set([
    'Wolverines', 'Hoosiers', 'Cyclones', 'Knights', 'Gators', 'Tigers',
    'Wolfpack', 'Dukes', 'Billikens', 'Bonnies', 'Buckeyes', 'Demon Deacons', 'Flashes', 'RedHawks', 'Ducks',
    'Spartans', 'Bears', 'Raiders', 'Razorbacks', 'Commodores', 'Bulldogs',
    'Bruins', 'Boilermakers', 'Buffaloes', 'Jayhawks', 'Wildcats', 'Aggies',
    'Huskies', 'Tar Heels', 'Blue Devils', 'Cardinals', 'Sooners', 'Longhorns',
    'Crimson Tide', 'Volunteers', 'Gamecocks', 'Rebels', 'Broncos', 'Cougars',
    'Panthers', 'Eagles', 'Owls', 'Rams', 'Bulls', 'Golden Knights', 'Mean Green',
    'Thundering Herd', 'Miners', 'Roadrunners', 'Hilltoppers', 'Golden Flashes',
    'Bearcats', 'Fighting Illini', 'Terrapins', 'Cornhuskers', 'Waves', 'Golden Gophers',
    'Cavaliers', 'Mountaineers', 'Hokies', 'Cowboys', 'Utes', 'Dons', 'Dolphins', 'Red Flash',
    'Chargers', 'Skyhawks', 'Lakers', 'Mastodons', 'Jaguars', 'Seahawks', 'Sharks', 'Salukis',
    'Purple Aces', 'Trojans', 'Badgers', 'Scarlet Knights', 'Friars', 'Revolutionaries', 'Minutemen', 'Horned Frogs', 'Flyers'
])

multi_word_mascots = ['Tar Heels', 'Blue Devils', 'Fighting Irish', 'Golden Flashes', 'Red Raiders',
                      'Golden Knights', 'Thundering Herd', 'Crimson Tide', 'Mean Green', 'Fighting Illini',
                      'Demon Deacons', 'Golden Gophers', 'Yellow Jackets', 'Red Flash', 'Purple Aces', 'Scarlet Knights']

special_cases = {
    'Miami (FL)': 'Miami',
    'Miami (OH)': 'Miami (OH)',
    'NC State': 'North Carolina State',
    'Kent State Golden Flashes': 'Kent State',
    'Texas Tech Red Raiders': 'Texas Tech',
    'IU Indianapolis Jaguars': 'IUPUI',
    'Long Island University Sharks': 'LIU',
    'Purdue Fort Wayne Mastodons': 'Purdue Fort Wayne',
    'Central Connecticut Blue Devils': 'Central Connecticut',
    'Chicago State Cougars': 'Chicago State',
    'Southern Illinois Salukis': 'Southern Illinois',
    'Saint Francis Red Flash': 'Saint Francis (PA)',
    'New Haven Chargers': 'Sacred Heart',
    'Stonehill Skyhawks': 'Stonehill',
    'Mercyhurst Lakers': 'Mercyhurst',
}


def normalize_team_name(name: str) -> str:
    if pd.isna(name):
        return ''
    name = name.strip()
    if name in special_cases:
        return special_cases[name]
    for mw in multi_word_mascots:
        if name.endswith(' ' + mw):
            return name[:-len(' ' + mw)].strip()
    parts = name.split()
    if len(parts) > 1 and parts[-1] in mascots:
        return ' '.join(parts[:-1])
    # common replacements
    name = name.replace('St.', 'St').replace('.', '')
    name = re.sub(r"\s+", " ", name)
    return name


def fuzzy_match(name, candidates):
    name_clean = name.lower()
    matches = difflib.get_close_matches(name_clean, [c.lower() for c in candidates], n=3, cutoff=0.8)
    return matches


def main():
    espn = pd.read_csv('data_files/espn_cbb_current_season.csv')
    kenpom = pd.read_csv('data_files/kenpom_ratings.csv')
    bart = pd.read_csv('data_files/barttorvik_ratings.csv')
    training = pd.read_csv('data_files/training_data_comprehensive.csv')

    # Build canonical set from training data (broader set of teams) using normalization
    training_teams = set()
    for col in ['home_team', 'away_team']:
        for name in training[col].unique():
            training_teams.add(normalize_team_name(name))

    print(f'Found {len(training_teams)} unique canonical training teams')

    # Also keep the (smaller) ESPN set for cross-checks
    espn_teams = set()
    for col in ['home_team', 'away_team']:
        for name in espn[col].unique():
            espn_teams.add(normalize_team_name(name))

    print(f'Found {len(espn_teams)} unique canonical ESPN teams')

    kenpom_teams = list(kenpom['Team'].astype(str).unique())
    
    # Handle BartTorvik data format (team names as first column values)
    bart_team_col = bart.columns[0]  # First column contains team names
    bart_teams = list(bart[bart_team_col].astype(str).unique())
    
    print(f'KenPom teams: {len(kenpom_teams)}; BartTorvik teams: {len(bart_teams)}')

    # Attempt to match KenPom teams
    kenpom_matches = {}
    kenpom_missing = []
    for t in kenpom_teams:
        t_norm = normalize_team_name(t)
        # common kenpom variants
        if t_norm.endswith(' FL'):
            t_norm = t_norm.replace(' FL', '')
        if t_norm == 'Connecticut':
            t_norm = 'UConn'

        if t_norm in training_teams:
            kenpom_matches[t] = t_norm
        else:
            # fuzzy match to training set
            candidates = difflib.get_close_matches(t_norm, list(training_teams), n=1, cutoff=0.75)
            if candidates:
                kenpom_matches[t] = candidates[0] + ' (fuzzy)'
            else:
                kenpom_matches[t] = ''
                kenpom_missing.append(t)

    # BartTorvik matches
    bart_matches = {}
    bart_missing = []
    for t in bart_teams:
        t_norm = normalize_team_name(t)
        if t_norm.endswith(' FL'):
            t_norm = t_norm.replace(' FL', '')
        if t_norm == 'Connecticut':
            t_norm = 'UConn'

        if t_norm in training_teams:
            bart_matches[t] = t_norm
        else:
            candidates = difflib.get_close_matches(t_norm, list(training_teams), n=1, cutoff=0.75)
            if candidates:
                bart_matches[t] = candidates[0] + ' (fuzzy)'
            else:
                bart_matches[t] = ''
                bart_missing.append(t)


    print('\nKenPom missing teams count:', len(kenpom_missing))
    print('\nKenPom missing sample:', kenpom_missing[:20])
    print('\nBartTorvik missing teams count:', len(bart_missing))
    print('\nBartTorvik missing sample:', bart_missing[:20])

    # Summarize match quality
    def summarize(matches):
        exact = sum(1 for v in matches.values() if v and '(fuzzy)' not in v)
        fuzzy = sum(1 for v in matches.values() if v and '(fuzzy)' in v)
        missing = sum(1 for v in matches.values() if not v)
        return exact, fuzzy, missing

    kp_exact, kp_fuzzy, kp_missing = summarize(kenpom_matches)
    bt_exact, bt_fuzzy, bt_missing = summarize(bart_matches)

    print(f"\nKenPom: exact={kp_exact}, fuzzy={kp_fuzzy}, missing={kp_missing} (total {len(kenpom_teams)})")
    print('\nKenPom missing sample:', kenpom_missing[:20])

    print(f"\nBartTorvik: exact={bt_exact}, fuzzy={bt_fuzzy}, missing={bt_missing} (total {len(bart_teams)})")
    print('\nBartTorvik missing sample:', bart_missing[:20])

    # Also report teams in ESPN not covered by either dataset
    matched_espn = set(v for v in kenpom_matches.values() if isinstance(v, str)) | set(v for v in bart_matches.values() if isinstance(v, str))
    espn_not_covered = sorted([t for t in espn_teams if t not in matched_espn])
    print(f'\nESPN teams not covered by either source: {len(espn_not_covered)}')
    print(espn_not_covered[:40])

    # Save match outputs
    pd.DataFrame([{'kenpom':k, 'espn_match':v if isinstance(v,str) else ','.join(v)} for k,v in kenpom_matches.items()])\
        .to_csv('data_files/kenpom_to_espn_matches.csv', index=False)
    pd.DataFrame([{'bart':k, 'espn_match':v if isinstance(v,str) else ','.join(v)} for k,v in bart_matches.items()])\
        .to_csv('data_files/bart_to_espn_matches.csv', index=False)


if __name__ == '__main__':
    main()
