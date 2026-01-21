# Feature Engineering for March Madness Betting Predictions

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import json
import os
from pathlib import Path

# Data directory
DATA_DIR = Path("data_files")
CACHE_DIR = DATA_DIR / "cache"

def load_cached_data(filename: str) -> Optional[List]:
    """Load data from cache if it exists."""
    cache_path = CACHE_DIR / f"{filename}.json"
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            return json.load(f)
    return None

def extract_team_stats(team_stats: dict) -> dict:
    """Extract relevant stats from team stats dictionary."""
    if not team_stats:
        return {
            "ppg": 0, "pace": 0, "efg_pct": 0, "to_rate": 0, "orb_pct": 0, "ft_rate": 0,
            "fg_pct": 0, "three_pct": 0, "opp_ppg": 0
        }

    games = team_stats.get("games", 1)  # Avoid division by zero
    team_stats_data = team_stats.get("teamStats", {})
    opp_stats_data = team_stats.get("opponentStats", {})

    # Safely extract values with defaults
    efg_pct = team_stats_data.get("fourFactors", {}).get("effectiveFieldGoalPct")
    efg_pct = efg_pct / 100 if efg_pct is not None else 0

    to_rate = team_stats_data.get("fourFactors", {}).get("turnoverRatio", 0)
    orb_pct = team_stats_data.get("fourFactors", {}).get("offensiveReboundPct")
    orb_pct = orb_pct / 100 if orb_pct is not None else 0

    ft_rate = team_stats_data.get("fourFactors", {}).get("freeThrowRate")
    ft_rate = ft_rate / 100 if ft_rate is not None else 0

    fg_pct = team_stats_data.get("fieldGoals", {}).get("pct")
    fg_pct = fg_pct / 100 if fg_pct is not None else 0

    three_pct = team_stats_data.get("threePointFieldGoals", {}).get("pct")
    three_pct = three_pct / 100 if three_pct is not None else 0

    return {
        "ppg": team_stats_data.get("points", {}).get("total", 0) / games,
        "pace": team_stats.get("pace", 0),
        "efg_pct": efg_pct,
        "to_rate": to_rate,
        "orb_pct": orb_pct,
        "ft_rate": ft_rate,
        "fg_pct": fg_pct,
        "three_pct": three_pct,
        "opp_ppg": opp_stats_data.get("points", {}).get("total", 0) / games,
    }

def calculate_efficiency_differential(team1_eff: dict, team2_eff: dict) -> dict:
    """Calculate efficiency differentials between two teams."""
    return {
        "off_eff_diff": team1_eff.get("offensiveRating", 0) - team2_eff.get("offensiveRating", 0),
        "def_eff_diff": team1_eff.get("defensiveRating", 0) - team2_eff.get("defensiveRating", 0),
        "net_eff_diff": team1_eff.get("netRating", 0) - team2_eff.get("netRating", 0)
    }

def calculate_spread_features(home_stats: dict, away_stats: dict, home_team_eff: dict, away_team_eff: dict) -> dict:
    """Features for predicting point spread."""
    return {
        # Efficiency-based
        "net_rating_diff": home_team_eff.get("netRating", 0) - away_team_eff.get("netRating", 0),
        "off_rating_diff": home_team_eff.get("offensiveRating", 0) - away_team_eff.get("offensiveRating", 0),
        "def_rating_diff": home_team_eff.get("defensiveRating", 0) - away_team_eff.get("defensiveRating", 0),

        # Scoring-based
        "ppg_diff": home_stats.get("ppg", 0) - away_stats.get("ppg", 0),
        "opp_ppg_diff": home_stats.get("opp_ppg", 0) - away_stats.get("opp_ppg", 0),
        "margin_diff": (home_stats.get("ppg", 0) - home_stats.get("opp_ppg", 0)) - (away_stats.get("ppg", 0) - away_stats.get("opp_ppg", 0)),

        # Four factors differentials
        "efg_diff": home_stats.get("efg_pct", 0) - away_stats.get("efg_pct", 0),
        "to_rate_diff": home_stats.get("to_rate", 0) - away_stats.get("to_rate", 0),
        "orb_diff": home_stats.get("orb_pct", 0) - away_stats.get("orb_pct", 0),
        "ft_rate_diff": home_stats.get("ft_rate", 0) - away_stats.get("ft_rate", 0),
    }

def calculate_total_features(home_stats: dict, away_stats: dict, home_team_eff: dict, away_team_eff: dict) -> dict:
    """Features for predicting game total."""
    # Calculate projected total first
    avg_tempo = (home_stats.get("pace", 70) + away_stats.get("pace", 70)) / 2
    avg_off_eff = (home_team_eff.get("offensiveRating", 100) + away_team_eff.get("offensiveRating", 100)) / 2
    avg_def_eff = (home_team_eff.get("defensiveRating", 100) + away_team_eff.get("defensiveRating", 100)) / 2
    projected_total = (avg_off_eff + avg_def_eff) / 2 * (avg_tempo / 100) * 0.8

    return {
        # Efficiency-based projections
        "combined_off_eff": home_team_eff.get("offensiveRating", 0) + away_team_eff.get("offensiveRating", 0),
        "combined_def_eff": home_team_eff.get("defensiveRating", 0) + away_team_eff.get("defensiveRating", 0),
        "avg_off_eff": (home_team_eff.get("offensiveRating", 0) + away_team_eff.get("offensiveRating", 0)) / 2,
        "avg_def_eff": (home_team_eff.get("defensiveRating", 0) + away_team_eff.get("defensiveRating", 0)) / 2,

        # Pace and scoring
        "combined_tempo": home_stats.get("pace", 0) + away_stats.get("pace", 0),
        "avg_tempo": (home_stats.get("pace", 0) + away_stats.get("pace", 0)) / 2,
        "combined_ppg": home_stats.get("ppg", 0) + away_stats.get("ppg", 0),
        "combined_opp_ppg": home_stats.get("opp_ppg", 0) + away_stats.get("opp_ppg", 0),

        # Shooting
        "combined_fg_pct": home_stats.get("fg_pct", 0) + away_stats.get("fg_pct", 0),
        "combined_3pt_pct": home_stats.get("three_pct", 0) + away_stats.get("three_pct", 0),

        # Projected total
        "projected_total": projected_total,
    }

def create_game_features(game: dict, efficiency_data: List[dict], team_stats_lookup: dict = None) -> Optional[dict]:
    """Create feature set for a single game."""
    try:
        # Find efficiency ratings for both teams
        home_team_eff = None
        away_team_eff = None

        for eff in efficiency_data:
            if eff.get("team") == game.get("homeTeam"):
                home_team_eff = eff
            elif eff.get("team") == game.get("awayTeam"):
                away_team_eff = eff

        if not home_team_eff or not away_team_eff:
            return None  # Skip games where we don't have efficiency data

        # Get team stats
        home_stats = team_stats_lookup.get(game.get("homeTeam"), {}) if team_stats_lookup else {}
        away_stats = team_stats_lookup.get(game.get("awayTeam"), {}) if team_stats_lookup else {}

        # Basic game info
        features = {
            "game_id": game.get("id"),
            "season": game.get("season"),
            "season_type": game.get("seasonType"),
            "home_team": game.get("homeTeam"),
            "away_team": game.get("awayTeam"),
            "home_score": game.get("homePoints"),
            "away_score": game.get("awayPoints"),
            "actual_spread": (game.get("awayPoints", 0) - game.get("homePoints", 0)) if game.get("homePoints") and game.get("awayPoints") else None,
            "actual_total": (game.get("homePoints", 0) + game.get("awayPoints", 0)) if game.get("homePoints") and game.get("awayPoints") else None,
        }

        # Betting lines (if available)
        lines = game.get("lines", [])
        if lines:
            line = lines[0]  # Use first available line
            features.update({
                "betting_spread": line.get("spread"),
                "betting_over_under": line.get("overUnder"),
                "home_moneyline": line.get("homeMoneyline"),
                "away_moneyline": line.get("awayMoneyline"),
            })

        # Efficiency features
        eff_features = calculate_efficiency_differential(home_team_eff, away_team_eff)
        features.update(eff_features)

        # Spread prediction features
        spread_features = calculate_spread_features(home_stats, away_stats, home_team_eff, away_team_eff)
        features.update({f"spread_{k}": v for k, v in spread_features.items()})

        # Total prediction features
        total_features = calculate_total_features(home_stats, away_stats, home_team_eff, away_team_eff)
        features.update({f"total_{k}": v for k, v in total_features.items()})

        return features

    except Exception as e:
        print(f"Error processing game {game.get('gameId')}: {e}")
        return None

def build_weighted_training_dataset(start_year: int = 2016, end_year: int = 2025, tournament_weight: float = 5.0) -> pd.DataFrame:
    """Build weighted training dataset combining regular season and tournament games."""
    print(f"Building weighted training dataset from {start_year} to {end_year}...")
    print(f"Tournament games weighted {tournament_weight}x more than regular season games")

    all_features = []

    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")

        # Load efficiency data (use most recent)
        efficiency = load_cached_data("efficiency_2024")

        if not efficiency:
            print(f"  Skipping {year} - no efficiency data")
            continue

        # Create team stats lookup
        team_stats = load_cached_data(f"team_stats_{year}")
        team_stats_lookup = {}
        if team_stats:
            for team in team_stats:
                team_name = team.get("team")
                if team_name:
                    team_stats_lookup[team_name] = extract_team_stats(team)

        # Process regular season games (limit to keep dataset manageable)
        reg_games = load_cached_data(f"games_{year}_regular")
        reg_count = 0
        if reg_games:
            for game in reg_games[:500]:  # Sample 500 regular season games per year
                game['season'] = year
                game['seasonType'] = 'regular'
                features = create_game_features(game, efficiency, team_stats_lookup)
                if features:
                    features['sample_weight'] = 1.0  # Regular season weight
                    features['game_type'] = 'regular'
                    all_features.append(features)
                    reg_count += 1

        # Process tournament games
        tour_games = load_cached_data(f"games_{year}_postseason")
        tour_count = 0
        if tour_games:
            for game in tour_games:
                game['season'] = year
                game['seasonType'] = 'postseason'
                features = create_game_features(game, efficiency, team_stats_lookup)
                if features:
                    features['sample_weight'] = tournament_weight  # Tournament weight
                    features['game_type'] = 'tournament'
                    all_features.append(features)
                    tour_count += 1

        print(f"  Added {reg_count} regular season, {tour_count} tournament games")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    if not df.empty:
        print(f"✅ Weighted dataset created: {len(df)} games, {len(df.columns)} features")
        print(f"   Years covered: {sorted(df['season'].unique())}")
        print(f"   Game types: {df['game_type'].value_counts().to_dict()}")
        print(f"   Weight distribution: Regular={1.0}, Tournament={tournament_weight}")
        print(f"   Total effective weight: {df['sample_weight'].sum():.1f}")

        # Save to CSV
        output_path = DATA_DIR / "training_data_weighted.csv"
        df.to_csv(output_path, index=False)
        print(f"   Saved to {output_path}")

    return df

def build_tournament_training_dataset(start_year: int = 2016, end_year: int = 2025) -> pd.DataFrame:
    """Build training dataset specifically from tournament games."""
    print(f"Building tournament training dataset from {start_year} to {end_year}...")

    all_features = []

    for year in range(start_year, end_year + 1):
        print(f"Processing tournament {year}...")

        # Load tournament games and betting lines
        tournament_games = load_cached_data(f"games_{year}_postseason")
        betting_lines = load_cached_data(f"lines_{year}_postseason")
        efficiency = load_cached_data("efficiency_2024")  # Use most recent efficiency ratings
        team_stats = load_cached_data(f"team_stats_{year}")

        if not tournament_games:
            print(f"  Skipping {year} - no tournament games")
            continue

        # Create team stats lookup dictionary
        team_stats_lookup = {}
        if team_stats:
            for team in team_stats:
                team_name = team.get("team")
                if team_name:
                    team_stats_lookup[team_name] = extract_team_stats(team)

        # Create betting lines lookup by game ID
        lines_lookup = {}
        if betting_lines:
            for line in betting_lines:
                game_id = line.get("gameId") or line.get("id")
                if game_id:
                    lines_lookup[game_id] = line

        print(f"  Loaded {len(tournament_games)} tournament games, {len(lines_lookup)} betting lines, stats for {len(team_stats_lookup)} teams")

        # Create features for each tournament game
        year_features = []
        for game in tournament_games:
            game_id = game.get("id") or game.get("gameId")
            features = create_game_features(game, efficiency, team_stats_lookup)

            # Add betting line data if available
            if game_id and game_id in lines_lookup:
                line = lines_lookup[game_id]
                features.update({
                    "betting_spread": line.get("spread"),
                    "betting_over_under": line.get("overUnder"),
                    "home_moneyline": line.get("homeMoneyline"),
                    "away_moneyline": line.get("awayMoneyline"),
                })

            if features:
                year_features.append(features)

        all_features.extend(year_features)
        print(f"  Added {len(year_features)} tournament games from {year}")

    # Convert to DataFrame
    df = pd.DataFrame(all_features)

    if not df.empty:
        print(f"✅ Tournament dataset created: {len(df)} games, {len(df.columns)} features")
        print(f"   Years covered: {sorted(df['season'].unique())}")
        print(f"   Games with betting data: {df['betting_spread'].notna().sum()}")

        # Save to CSV
        output_path = DATA_DIR / "training_data_tournament.csv"
        df.to_csv(output_path, index=False)
        print(f"   Saved to {output_path}")

    return df

if __name__ == "__main__":
    # Build weighted training dataset (regular season + tournament with higher tournament weight)
    print("🏀 Building weighted training dataset (2016-2025)...")
    print("Regular season + Tournament games (tournament weighted 5x more)")

    df = build_weighted_training_dataset(2016, 2025, tournament_weight=5.0)

    if not df.empty:
        print("\n✅ Weighted dataset summary:")
        print(f"   Total games: {len(df)}")
        print(f"   Years: {sorted(df['season'].unique())}")
        print(f"   Features: {len(df.columns)}")
        print(f"   Game breakdown: {df['game_type'].value_counts().to_dict()}")
        print(f"   Weight distribution: Regular=1.0, Tournament=5.0")
        print(f"   Effective sample size: {df['sample_weight'].sum():.0f} (vs raw {len(df)})")
        print(f"   Games with results: {df['actual_spread'].notna().sum()}")

        # Show sample
        print("\nSample games:")
        sample_reg = df[df['game_type'] == 'regular'].head(2)
        sample_tour = df[df['game_type'] == 'tournament'].head(2)
        for _, row in pd.concat([sample_reg, sample_tour]).iterrows():
            weight = row.get('sample_weight', 1.0)
            spread = row.get('actual_spread', 'N/A')
            print(f"  {row['game_type'].title()}: {row['home_team']} vs {row['away_team']} (weight: {weight}, spread: {spread})")
    else:
        print("❌ Failed to create weighted dataset")