from predictions import normalize_team_name
from fetch_live_odds import fetch_live_odds

# Test the normalization
home_team = 'Kansas State Wildcats'
away_team = 'Kansas Jayhawks'

home_norm = normalize_team_name(home_team)
away_norm = normalize_team_name(away_team)

print(f'ESPN home_team: "{home_team}" -> "{home_norm}"')
print(f'ESPN away_team: "{away_team}" -> "{away_norm}"')

odds = fetch_live_odds()
kansas_keys = [k for k in odds.keys() if 'Kansas' in k]
print(f'Odds keys with Kansas: {kansas_keys}')

# Check what the odds API actually has
for key in kansas_keys:
    parts = key.split(' vs ')
    odds_home = parts[0]
    odds_away = parts[1]
    print(f'Odds: home="{odds_home}", away="{odds_away}"')