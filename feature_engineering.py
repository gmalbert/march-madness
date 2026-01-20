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

    return {
        "ppg": team_stats_data.get("points", {}).get("total", 0) / games,
        "pace": team_stats.get("pace", 0),
        "efg_pct": team_stats_data.get("fourFactors", {}).get("effectiveFieldGoalPct", 0) / 100,  # Convert to decimal
        "to_rate": team_stats_data.get("fourFactors", {}).get("turnoverRatio", 0),
        "orb_pct": team_stats_data.get("fourFactors", {}).get("offensiveReboundPct", 0) / 100,
        "ft_rate": team_stats_data.get("fourFactors", {}).get("freeThrowRate", 0) / 100,
        "fg_pct": team_stats_data.get("fieldGoals", {}).get("pct", 0) / 100,
        "three_pct": team_stats_data.get("threePointFieldGoals", {}).get("pct", 0) / 100,
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

def build_training_dataset(year: int) -> pd.DataFrame:
    """Build complete training dataset for a given year."""
    print(f"Building training dataset for {year}...")

    # Load data
    games = load_cached_data(f"games_{year}_regular")
    # Use most recent efficiency ratings (they're updated throughout the season)
    efficiency = load_cached_data("efficiency_2024")
    team_stats = load_cached_data(f"team_stats_{year}")

    if not games or not efficiency:
        print(f"Missing data for {year} - games: {len(games) if games else 0}, efficiency: {len(efficiency) if efficiency else 0}")
        return pd.DataFrame()

    # Create team stats lookup dictionary
    team_stats_lookup = {}
    if team_stats:
        for team in team_stats:
            team_name = team.get("team")
            if team_name:
                team_stats_lookup[team_name] = extract_team_stats(team)

    print(f"Loaded stats for {len(team_stats_lookup)} teams")

    # Create features for each game
    feature_list = []
    for game in games:
        features = create_game_features(game, efficiency, team_stats_lookup)
        if features:
            feature_list.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(feature_list)

    print(f"Created dataset with {len(df)} games and {len(df.columns)} features")

    return df

if __name__ == "__main__":
    # Test feature engineering
    print("🧪 Testing feature engineering...")

    # Build dataset for 2023
    df = build_training_dataset(2023)

    if not df.empty:
        print(f"✅ Dataset created: {df.shape}")
        print("Sample features:")
        print(df.head())

        # Save to CSV
        output_path = DATA_DIR / "training_data_2023.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved to {output_path}")
    else:
        print("❌ Failed to create dataset")