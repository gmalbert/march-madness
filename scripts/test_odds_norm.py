from fetch_live_odds import normalize_team_name

# Test the normalization on the actual Odds API team names
test_names = ['Kansas St Wildcats', 'Kansas Jayhawks']
for name in test_names:
    normalized = normalize_team_name(name)
    print(f'Odds API: "{name}" -> "{normalized}"')