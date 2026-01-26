#!/usr/bin/env python3
"""
Create historical games with betting data for model evaluation.

This script combines tournament games with betting lines, runs model predictions
on historical data, and saves the results for betting model evaluation.
"""

import sys
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List
import joblib

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from data_collection import load_cached
from predictions import calculate_features, make_predictions
from data_tools.efficiency_loader import EfficiencyDataLoader

DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"

def load_models() -> Dict:
    """Load trained models."""
    models = {}
    model_files = {
        'spread': 'spread_model.pkl',
        'total': 'total_model.pkl',
        'moneyline': 'moneyline_model.pkl'
    }

    for model_type, filename in model_files.items():
        model_path = MODEL_DIR / filename
        if model_path.exists():
            models[model_type] = joblib.load(model_path)
            print(f"Loaded {model_type} model")
        else:
            print(f"Warning: {model_type} model not found at {model_path}")

    return models

def create_historical_betting_dataset():
    """Create dataset of historical games with betting data and model predictions."""

    print("Loading tournament games and betting lines...")

    # Load tournament games
    tournament_games = []
    for year in range(2016, 2026):
        games = load_cached(f"games_{year}_postseason")
        if games:
            for game in games:
                if isinstance(game, dict):
                    game['season_year'] = year
                else:
                    game_dict = game.to_dict() if hasattr(game, 'to_dict') else dict(game.__dict__)
                    game_dict['season_year'] = year
                    game = game_dict
            tournament_games.extend(games)

    print(f"Loaded {len(tournament_games)} tournament games")

    # Load betting lines
    betting_lines = []
    for year in range(2016, 2026):
        lines = load_cached(f"lines_{year}_postseason")
        if lines:
            for line in lines:
                if isinstance(line, dict):
                    line['season_year'] = year
                else:
                    line_dict = line.to_dict() if hasattr(line, 'to_dict') else dict(line.__dict__)
                    line_dict['season_year'] = year
                    line = line_dict
            betting_lines.extend(lines)

    print(f"Loaded {len(betting_lines)} betting lines")

    # Load efficiency data
    loader = EfficiencyDataLoader()
    kenpom_df = loader.load_kenpom()
    bart_df = loader.load_barttorvik()

    print("Merging games with betting lines...")

    # Create lookup for betting lines by game ID
    lines_lookup = {}
    for line_data in betting_lines:
        game_id = line_data.get('id') or line_data.get('gameId')
        if game_id:
            lines_lookup[game_id] = line_data

    # Load models
    models = load_models()

    historical_data = []

    for game in tournament_games:
        game_id = game.get('id') or game.get('gameId')
        line_data = lines_lookup.get(game_id)

        if not line_data or not line_data.get('lines'):
            continue  # Skip games without betting lines

        # Get the first (most recent) line
        line = line_data['lines'][0] if isinstance(line_data['lines'], list) else line_data['lines']

        # Extract game info
        home_team = game.get('homeTeam') or game.get('home_team')
        away_team = game.get('awayTeam') or game.get('away_team')
        home_score = game.get('homePoints') or game.get('home_score')
        away_score = game.get('awayPoints') or game.get('away_score')

        if not all([home_team, away_team, home_score is not None, away_score is not None]):
            continue  # Skip incomplete games

        # Calculate actual results
        home_win = 1 if home_score > away_score else 0
        actual_spread = away_score - home_score  # Positive means home covers
        actual_total = home_score + away_score

        # Extract betting data
        spread = line.get('spread', 0)
        over_under = line.get('overUnder', 0)
        home_moneyline = line.get('homeMoneyline')
        away_moneyline = line.get('awayMoneyline')

        if spread is None:
            spread = 0
        if over_under is None:
            over_under = 0

        # Determine ATS result (1 if home covers, 0 if away covers)
        ats_result = 1 if actual_spread > spread else 0

        # Determine over/under result (1 if over hits, 0 if under hits)
        over_result = 1 if actual_total > over_under else 0

        # Get team stats (simplified - using basic efficiency)
        home_eff = {'offensiveRating': 100, 'defensiveRating': 100, 'netRating': 0}
        away_eff = {'offensiveRating': 100, 'defensiveRating': 100, 'netRating': 0}

        # Get advanced metrics
        advanced_metrics = {'home': {}, 'away': {}}

        # Add KenPom data if available
        if kenpom_df is not None:
            home_kp = kenpom_df[kenpom_df['canonical_team'] == home_team]
            away_kp = kenpom_df[kenpom_df['canonical_team'] == away_team]

            if len(home_kp) > 0:
                advanced_metrics['home']['kenpom'] = home_kp.iloc[0].to_dict()
            if len(away_kp) > 0:
                advanced_metrics['away']['kenpom'] = away_kp.iloc[0].to_dict()

        # Add BartTorvik data if available
        if bart_df is not None:
            home_bt = bart_df[bart_df['canonical_team'] == home_team]
            away_bt = bart_df[bart_df['canonical_team'] == away_team]

            if len(home_bt) > 0:
                advanced_metrics['home']['barttorvik'] = home_bt.iloc[0].to_dict()
            if len(away_bt) > 0:
                advanced_metrics['away']['barttorvik'] = away_bt.iloc[0].to_dict()

        # Create game data structure for prediction
        game_data = {
            'home_stats': {},  # Simplified
            'away_stats': {},  # Simplified
            'home_eff': home_eff,
            'away_eff': away_eff
        }

        # Make predictions
        try:
            predictions = make_predictions(game_data, models, advanced_metrics)
        except Exception as e:
            print(f"Error making predictions for {home_team} vs {away_team}: {e}")
            continue

        # Extract prediction results
        pred_home_win_prob = predictions.get('moneyline', {}).get('probability', 0.5)
        pred_spread = predictions.get('spread', {}).get('prediction', 0)
        pred_total = predictions.get('total', {}).get('prediction', 0)

        # Create data row
        row = {
            'season': game.get('season_year', 2024),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'home_win': home_win,
            'actual_spread': actual_spread,
            'actual_total': actual_total,
            'betting_spread': spread,
            'betting_over_under': over_under,
            'home_moneyline': home_moneyline,
            'away_moneyline': away_moneyline,
            'ats_result': ats_result,
            'over_result': over_result,
            'pred_home_win_prob': pred_home_win_prob,
            'pred_spread': pred_spread,
            'pred_total': pred_total
        }

        historical_data.append(row)

    # Convert to DataFrame and save
    df = pd.DataFrame(historical_data)
    output_path = DATA_DIR / "historical_games_with_betting.csv"
    df.to_csv(output_path, index=False)

    print(f"Saved {len(df)} historical games with betting data to {output_path}")
    print(f"Years covered: {sorted(df['season'].unique())}")
    print(f"Games with moneyline odds: {df['home_moneyline'].notna().sum()}")
    print(f"Games with spread odds: {(df['betting_spread'] != 0).sum()}")

if __name__ == "__main__":
    create_historical_betting_dataset()