# Analyze Missing Efficiency Data

import json
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path("data_files/cache")

def load_cached_data(filename: str):
    """Load data from cache."""
    cache_path = DATA_DIR / f"{filename}.json"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

# Load efficiency data
efficiency = load_cached_data("efficiency_2024")
teams_in_efficiency = set()
if efficiency:
    for eff in efficiency:
        if eff and eff.get("team"):
            teams_in_efficiency.add(eff.get("team"))

print(f"Total teams with efficiency data: {len(teams_in_efficiency)}")

# Analyze by year
stats_by_year = defaultdict(lambda: {
    'total_games': 0,
    'missing_home': 0,
    'missing_away': 0,
    'missing_both': 0,
    'missing_teams': set()
})

for year in range(2016, 2026):
    # Load games
    regular_games = load_cached_data(f"games_{year}_regular") or []
    tournament_games = load_cached_data(f"games_{year}_postseason") or []
    
    all_games = []
    if regular_games:
        all_games.extend([g for g in regular_games if g is not None and isinstance(g, dict)])
    if tournament_games:
        all_games.extend([g for g in tournament_games if g is not None and isinstance(g, dict)])
    
    for game in all_games:
        if not isinstance(game, dict):
            continue
            
        stats_by_year[year]['total_games'] += 1
        
        home_team = game.get('homeTeam')
        away_team = game.get('awayTeam')
        
        has_home = home_team in teams_in_efficiency
        has_away = away_team in teams_in_efficiency
        
        if not has_home and not has_away:
            stats_by_year[year]['missing_both'] += 1
            if home_team:
                stats_by_year[year]['missing_teams'].add(home_team)
            if away_team:
                stats_by_year[year]['missing_teams'].add(away_team)
        elif not has_home:
            stats_by_year[year]['missing_home'] += 1
            if home_team:
                stats_by_year[year]['missing_teams'].add(home_team)
        elif not has_away:
            stats_by_year[year]['missing_away'] += 1
            if away_team:
                stats_by_year[year]['missing_teams'].add(away_team)

# Print statistics
print("\n" + "="*80)
print("MISSING EFFICIENCY DATA BY YEAR")
print("="*80)

total_games = 0
total_skipped = 0

for year in sorted(stats_by_year.keys()):
    stats = stats_by_year[year]
    skipped = stats['missing_home'] + stats['missing_away'] + stats['missing_both']
    total_games += stats['total_games']
    total_skipped += skipped
    
    pct = (skipped / stats['total_games'] * 100) if stats['total_games'] > 0 else 0
    
    print(f"\n{year}:")
    print(f"  Total games: {stats['total_games']:,}")
    print(f"  Games skipped: {skipped:,} ({pct:.1f}%)")
    print(f"    - Missing home only: {stats['missing_home']:,}")
    print(f"    - Missing away only: {stats['missing_away']:,}")
    print(f"    - Missing both: {stats['missing_both']:,}")
    print(f"  Unique teams missing: {len(stats['missing_teams'])}")

print("\n" + "="*80)
print("OVERALL SUMMARY")
print("="*80)
print(f"Total games across all years: {total_games:,}")
print(f"Total games skipped: {total_skipped:,} ({total_skipped/total_games*100:.1f}%)")
print(f"Games successfully processed: {total_games - total_skipped:,} ({(total_games-total_skipped)/total_games*100:.1f}%)")

# Find most common missing teams
all_missing_teams = set()
for stats in stats_by_year.values():
    all_missing_teams.update(stats['missing_teams'])

print(f"\nTotal unique teams missing efficiency data: {len(all_missing_teams)}")
print(f"\nSample of missing teams (first 20):")
for i, team in enumerate(sorted(list(all_missing_teams)[:20])):
    print(f"  - {team}")
if len(all_missing_teams) > 20:
    print(f"  ... and {len(all_missing_teams) - 20} more")

# Check what's in the dataset we created
print("\n" + "="*80)
print("ACTUAL DATASET CREATED")
print("="*80)

import pandas as pd
df = pd.read_csv('data_files/training_data_weighted.csv')
print(f"Total games in dataset: {len(df):,}")
print(f"Years: {sorted(df['season'].unique())}")
print(f"Games by season type:")
for season_type, count in df['season_type'].value_counts().items():
    print(f"  {season_type}: {count:,}")
print(f"\nGames with betting lines: {df['betting_spread'].notna().sum():,}")
