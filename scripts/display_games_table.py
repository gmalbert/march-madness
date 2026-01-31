#!/usr/bin/env python3
"""
Display all upcoming games in a formatted table
"""
import pandas as pd

# Conference mapping
conference_map = {
    1: 'AAC', 2: 'ACC', 3: 'A-10', 4: 'Big East', 5: 'Big Sky', 6: 'Big South',
    7: 'Big Ten', 8: 'Big 12', 9: 'CAA', 10: 'C-USA', 11: 'Horizon', 12: 'Ivy',
    13: 'MAAC', 14: 'MAC', 15: 'MEAC', 16: 'MVC', 17: 'NEC', 18: 'OVC',
    19: 'Pac-12', 20: 'Patriot', 21: 'SEC', 22: 'SoCon', 23: 'SEC', 24: 'Southland',
    25: 'Summit', 26: 'Sun Belt', 27: 'SWAC', 28: 'WAC', 29: 'WCC', 30: 'Independent',
    45: 'ASUN', 62: 'America East'
}

def main():
    # Load the games data
    df = pd.read_csv('data_files/espn_cbb_current_season.csv')

    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    print('🏀 MARCH MADNESS PREDICTION GAMES (50 Total)')
    print('=' * 110)
    print(f"{'Date':<12} {'Away Team':<28} {'Home Team':<28} {'Conference':<12} {'Rankings'}")
    print('=' * 110)

    for _, game in df.iterrows():
        date_str = game['date'].strftime('%m/%d %H:%M')
        away_team = str(game['away_team'])[:27]  # Truncate long names
        home_team = str(game['home_team'])[:27]
        conf_name = conference_map.get(int(game['home_conference']), f'Conf {game["home_conference"]}')

        # Rankings
        rankings = []
        if pd.notna(game.get('away_rank')) and game['away_rank'] > 0:
            rankings.append(f'Away #{int(game["away_rank"])}')
        if pd.notna(game.get('home_rank')) and game['home_rank'] > 0:
            rankings.append(f'Home #{int(game["home_rank"])}')
        rank_str = ', '.join(rankings) if rankings else ''

        print(f'{date_str:<12} {away_team:<28} {home_team:<28} {conf_name:<12} {rank_str}')

    print('=' * 110)
    print(f'Total Games: {len(df)}')
    print()
    print('Conference Summary:')
    conf_counts = df['home_conference'].value_counts()
    for conf_id, count in conf_counts.items():
        conf_name = conference_map.get(int(conf_id), f'Unknown ({conf_id})')
        print(f'  {conf_name}: {count} games')

if __name__ == "__main__":
    main()