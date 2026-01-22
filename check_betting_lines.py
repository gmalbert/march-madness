from data_collection import fetch_betting_lines

lines = fetch_betting_lines(2024, 'postseason')
print(f'Total games: {len(lines)}')

# Find ones with actual betting data
with_lines = [g for g in lines if g.get('lines') and len(g['lines']) > 0]
print(f'Games with betting lines: {len(with_lines)}')

if with_lines:
    sample = with_lines[0]
    print(f'\nSample game: {sample["homeTeam"]} vs {sample["awayTeam"]}')
    print(f'  Season: {sample.get("season")}')
    print(f'  Date: {sample.get("startDate")}')
    line = sample['lines'][0]
    print(f'  Spread: {line.get("spread", "N/A")}')
    print(f'  Over/Under: {line.get("overUnder", "N/A")}')
    print(f'  Provider: {line.get("provider", "N/A")}')
    
    # Show a few more examples
    print(f'\nShowing first 5 games with betting lines:')
    for i, game in enumerate(with_lines[:5]):
        line = game['lines'][0]
        print(f'{i+1}. {game["awayTeam"]} @ {game["homeTeam"]} - Spread: {line.get("spread")}, O/U: {line.get("overUnder")}')
