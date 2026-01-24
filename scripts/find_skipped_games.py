#!/usr/bin/env python3
"""Identify upcoming games skipped due to missing team data."""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import pandas as pd
import predictions


def main():
    efficiency_list, stats_list, season_used = predictions.get_team_data(2025)
    print('season_used', season_used)
    espn_df = pd.read_csv('data_files/espn_cbb_current_season.csv')
    espn_df['date_dt'] = pd.to_datetime(espn_df['date'])
    upcoming = espn_df[espn_df['date_dt'] > pd.Timestamp.now(tz='UTC')].copy()
    print('upcoming count', len(upcoming))
    skipped = []
    prepared = []
    for idx, row in upcoming.iterrows():
        enriched = predictions.enrich_espn_game_with_cbbd_data(row, efficiency_list, stats_list, season_used)
        if not enriched:
            home = row.get('home_team')
            away = row.get('away_team')
            # determine which data missing
            home_canon = predictions.normalize_team_name(home)
            away_canon = predictions.normalize_team_name(away)
            home_eff = next((e for e in efficiency_list if e.get('team') == home_canon), None)
            away_eff = next((e for e in efficiency_list if e.get('team') == away_canon), None)
            home_stats = next((s for s in stats_list if s.get('team') == home_canon), None)
            away_stats = next((s for s in stats_list if s.get('team') == away_canon), None)
            missing = []
            if not home_eff:
                missing.append('home_eff')
            if not away_eff:
                missing.append('away_eff')
            if not home_stats:
                missing.append('home_stats')
            if not away_stats:
                missing.append('away_stats')
            skipped.append((home, away, home_canon, away_canon, missing))
        else:
            prepared.append((enriched['home_team'], enriched['away_team']))
    print('prepared count', len(prepared))
    print('skipped count', len(skipped))
    for s in skipped:
        print(s)

if __name__ == '__main__':
    main()
