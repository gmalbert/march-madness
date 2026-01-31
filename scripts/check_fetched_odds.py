from fetch_live_odds import fetch_live_odds
odds = fetch_live_odds(force_refresh=True)
print(f'Fetched {len(odds)} games from Odds API')
print()

# Count how many have moneyline
with_moneyline = 0
for key, data in odds.items():
    if data.get('home_moneyline') and data.get('away_moneyline'):
        with_moneyline += 1
        if with_moneyline <= 10:  # Show first 10
            print(f'{key}: {data["home_moneyline"]} / {data["away_moneyline"]}')

print(f'\nTotal games with moneyline: {with_moneyline}/{len(odds)} ({with_moneyline/len(odds)*100:.1f}%)')