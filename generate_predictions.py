#!/usr/bin/env python3
"""
Generate predictions for upcoming college basketball games.
This script loads upcoming games, fetches team data, and generates predictions.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import fetch_team_stats, fetch_efficiency_ratings, fetch_adjusted_efficiency
from features import (
    calculate_efficiency_differential,
    calculate_spread_features,
    calculate_total_features,
    calculate_win_probability_features,
    project_game_total,
)

# Configuration
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"

def load_models():
    """Load trained prediction models."""
    models = {}

    # Load advanced models first (these are the primary models)
    advanced_models = ['moneyline_advanced', 'spread_advanced', 'total_advanced']
    for model_name in advanced_models:
        model_file = MODEL_DIR / f"{model_name}.joblib"
        if model_file.exists():
            try:
                model_key = model_name.replace('_advanced', '')
                models[f'{model_key}_advanced'] = joblib.load(model_file)
                print(f"Loaded advanced {model_key} model")
            except Exception as e:
                print(f"Error loading advanced {model_name}: {e}")

    # Load basic models as fallback (suppress warnings for old versions)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        model_types = ['spread', 'total', 'moneyline']
        for model_type in model_types:
            models[model_type] = {}
            models[f'{model_type}_scalers'] = {}

            # Try to load each model variant
            for variant in ['xgboost', 'random_forest', 'linear_regression', 'logistic_regression']:
                model_file = MODEL_DIR / f"{model_type}_{variant}.joblib"
                if model_file.exists():
                    try:
                        models[model_type][variant] = joblib.load(model_file)
                        print(f"Loaded {model_type} {variant} model")
                    except Exception as e:
                        print(f"Error loading {model_type} {variant}: {e}")

            # Load scalers for linear/logistic models
            scaler_file = MODEL_DIR / f"{model_type}_linear_regression_scaler.joblib"
            if scaler_file.exists():
                try:
                    models[f'{model_type}_scalers']['linear'] = joblib.load(scaler_file)
                    print(f"Loaded {model_type} linear scaler")
                except Exception as e:
                    print(f"Error loading {model_type} scaler: {e}")

    return models

def calculate_features(home_stats, away_stats, home_eff, away_eff):
    """Calculate prediction features from team data using `features.py` helpers.

    Falls back to the original minimal efficiency-diff features if richer
    fields aren't available.
    """
    # Try to compute efficiency differentials using features helper
    try:
        eff_diff = calculate_efficiency_differential(home_eff or {}, away_eff or {})
    except Exception:
        # Fallback to legacy keys
        home_off = (home_eff.get('adj_off') if home_eff else None) or (home_eff.get('offensiveRating') if home_eff else None) or 100
        away_off = (away_eff.get('adj_off') if away_eff else None) or (away_eff.get('offensiveRating') if away_eff else None) or 100
        home_def = (home_eff.get('adj_def') if home_eff else None) or (home_eff.get('defensiveRating') if home_eff else None) or 100
        away_def = (away_eff.get('adj_def') if away_eff else None) or (away_eff.get('defensiveRating') if away_eff else None) or 100
        eff_diff = {
            'off_eff_diff': float(home_off) - float(away_off),
            'def_eff_diff': float(home_def) - float(away_def),
            'net_eff_diff': (float(home_off) - float(home_def)) - (float(away_off) - float(away_def))
        }

    # Build spread/total/moneyline feature dicts using helpers when possible
    try:
        spread_feats = calculate_spread_features(home_stats or {}, away_stats or {}, home_eff or {}, away_eff or {})
    except Exception:
        spread_feats = {
            'net_rating_diff': eff_diff.get('net_eff_diff', 0),
            'off_rating_diff': eff_diff.get('off_eff_diff', 0),
            'def_rating_diff': eff_diff.get('def_eff_diff', 0),
            'ppg_diff': (home_stats or {}).get('ppg', 0) - (away_stats or {}).get('ppg', 0),
            'opp_ppg_diff': (home_stats or {}).get('opp_ppg', 0) - (away_stats or {}).get('opp_ppg', 0),
            'margin_diff': 0,
            'efg_diff': (home_stats or {}).get('efg_pct', 0) - (away_stats or {}).get('efg_pct', 0),
            'to_rate_diff': (home_stats or {}).get('to_rate', 0) - (away_stats or {}).get('to_rate', 0),
            'orb_diff': (home_stats or {}).get('orb_pct', 0) - (away_stats or {}).get('orb_pct', 0),
            'ft_rate_diff': (home_stats or {}).get('ft_rate', 0) - (away_stats or {}).get('ft_rate', 0),
        }

    try:
        total_feats = calculate_total_features(home_stats or {}, away_stats or {}, home_eff or {}, away_eff or {})
    except Exception:
        total_feats = {
            'combined_tempo': (home_stats or {}).get('pace', 70) + (away_stats or {}).get('pace', 70),
            'avg_tempo': ((home_stats or {}).get('pace', 70) + (away_stats or {}).get('pace', 70)) / 2,
            'combined_ppg': (home_stats or {}).get('ppg', 0) + (away_stats or {}).get('ppg', 0),
            'combined_opp_ppg': (home_stats or {}).get('opp_ppg', 0) + (away_stats or {}).get('opp_ppg', 0),
            'combined_off_eff': (home_eff or {}).get('offensiveRating', 0) + (away_eff or {}).get('offensiveRating', 0),
            'combined_def_eff': (home_eff or {}).get('defensiveRating', 0) + (away_eff or {}).get('defensiveRating', 0),
            'projected_total': project_game_total(home_eff or {}, away_eff or {})
        }

    # Moneyline features: reuse spread_feats + win-prob helpers when available
    try:
        win_feats = calculate_win_probability_features(home_stats or {}, away_stats or {})
    except Exception:
        win_feats = {'net_rating_diff': eff_diff.get('net_eff_diff', 0)}

    # Also include the original 3-feature vector for backward compatibility
    minimal = {
        'off_eff_diff': eff_diff.get('off_eff_diff', 0),
        'def_eff_diff': eff_diff.get('def_eff_diff', 0),
        'net_eff_diff': eff_diff.get('net_eff_diff', 0)
    }

    return {
        'spread': {**minimal, **spread_feats},
        'total': {**minimal, **total_feats},
        'moneyline': {**minimal, **win_feats}
    }

def make_predictions(game_data, models):
    """Make predictions for a game using trained models."""
    features = calculate_features(
        game_data.get('home_stats'), game_data.get('away_stats'),
        game_data.get('home_eff'), game_data.get('away_eff')
    )

    predictions = {}

    # Define feature names that match training data
    feature_names = ['off_eff_diff', 'def_eff_diff', 'net_eff_diff']

    # Use advanced models if available (prioritize these)
    feature_df = pd.DataFrame([features['spread']], columns=feature_names)

    # Advanced spread model
    if models.get('spread_advanced'):
        try:
            pred = models['spread_advanced'].predict(feature_df)[0]
            predictions['spread_prediction'] = float(pred)
        except Exception as e:
            print(f"Error with advanced spread model: {e}")

    # Advanced total model
    if models.get('total_advanced'):
        try:
            pred = models['total_advanced'].predict(feature_df)[0]
            predictions['total_prediction'] = float(pred)
        except Exception as e:
            print(f"Error with advanced total model: {e}")

    # Advanced moneyline model
    if models.get('moneyline_advanced'):
        try:
            pred_proba = models['moneyline_advanced'].predict_proba(feature_df)[0]
            home_win_prob = pred_proba[1]
            away_win_prob = pred_proba[0]
            predictions['moneyline_home_win_prob'] = float(home_win_prob)
            predictions['moneyline_away_win_prob'] = float(away_win_prob)
        except Exception as e:
            print(f"Error with advanced moneyline model: {e}")

    # Fallback to basic models only if advanced models not available
    if not models.get('spread_advanced') and models.get('spread'):
        spread_preds = []
        spread_df = pd.DataFrame([features['spread']], columns=feature_names)
        scalers = models.get('spread_scalers', {})

        for model_name, model in models['spread'].items():
            try:
                if model_name in scalers:
                    scaled_df = scalers[model_name].transform(spread_df)
                    pred = model.predict(scaled_df)[0]
                else:
                    pred = model.predict(spread_df)[0]
                spread_preds.append(pred)
            except Exception as e:
                print(f"Error predicting spread with {model_name}: {e}")

        if spread_preds:
            predictions['spread_prediction'] = float(np.mean(spread_preds))

    if not models.get('total_advanced') and models.get('total'):
        total_preds = []
        total_df = pd.DataFrame([features['total']], columns=feature_names)
        scalers = models.get('total_scalers', {})

        for model_name, model in models['total'].items():
            try:
                if model_name in scalers:
                    scaled_df = scalers[model_name].transform(total_df)
                    pred = model.predict(scaled_df)[0]
                else:
                    pred = model.predict(total_df)[0]
                total_preds.append(pred)
            except Exception as e:
                print(f"Error predicting total with {model_name}: {e}")

        if total_preds:
            predictions['total_prediction'] = float(np.mean(total_preds))

    if not models.get('moneyline_advanced') and models.get('moneyline'):
        moneyline_preds = []
        moneyline_df = pd.DataFrame([features['moneyline']], columns=feature_names)
        scalers = models.get('moneyline_scalers', {})

        for model_name, model in models['moneyline'].items():
            try:
                if model_name in scalers:
                    scaled_df = scalers[model_name].transform(moneyline_df)
                    pred_proba = model.predict_proba(scaled_df)[0]
                else:
                    pred_proba = model.predict_proba(moneyline_df)[0]
                home_win_prob = pred_proba[1]
                moneyline_preds.append(home_win_prob)
            except Exception as e:
                print(f"Error predicting moneyline with {model_name}: {e}")

        if moneyline_preds:
            avg_prob = np.mean(moneyline_preds)
            predictions['moneyline_home_win_prob'] = float(avg_prob)
            predictions['moneyline_away_win_prob'] = float(1 - avg_prob)

    return predictions

def load_team_data_once():
    """Load team efficiency and stats data once for all games."""
    current_year = 2025  # Use 2025 season data for current predictions

    print("Loading team data for all games...")
    efficiency_data = fetch_efficiency_ratings(current_year)
    team_stats_data = fetch_team_stats(current_year)

    # Create lookup dictionaries
    efficiency_lookup = {team['team']: team for team in efficiency_data}
    stats_lookup = {team['team']: team for team in team_stats_data}

    print(f"Loaded data for {len(efficiency_data)} teams")
    return efficiency_lookup, stats_lookup

def load_upcoming_games():
    """Load upcoming games from the current season CSV."""
    games_file = DATA_DIR / "espn_cbb_current_season.csv"
    if not games_file.exists():
        print(f"No upcoming games file found: {games_file}")
        return []

    df = pd.read_csv(games_file)

    # Filter to upcoming games (not completed)
    upcoming = df[df['status'] == 'STATUS_SCHEDULED'].copy()

    print(f"Found {len(upcoming)} upcoming games")

    return upcoming.to_dict('records')

def fetch_game_data(game, efficiency_lookup, stats_lookup):
    """Fetch team stats and efficiency data for a game using pre-loaded data."""
    current_year = 2025  # Use 2025 season data for current predictions

    try:
        # Get data for both teams
        home_team = game['home_team']
        away_team = game['away_team']

        # Clean team names for matching (remove common suffixes)
        def clean_team_name(name):
            return (name.replace(' Bonnies', '').replace(' Billikens', '')
                       .replace(' Wolverines', '').replace(' Buckeyes', ''))

        home_clean = clean_team_name(home_team)
        away_clean = clean_team_name(away_team)

        # Get efficiency data with fallbacks
        home_eff = efficiency_lookup.get(home_clean)
        away_eff = efficiency_lookup.get(away_clean)

        # If no efficiency data, create reasonable defaults based on rankings
        if not home_eff:
            home_rank = game.get('home_rank', 50)
            home_eff = {
                'adj_off': 110 - (home_rank / 10),  # Better teams have higher offensive rating
                'adj_def': 110 - (home_rank / 10)   # Better teams have better defense
            }
        else:
            home_eff = {
                'adj_off': home_eff.get('offensiveRating', 100),
                'adj_def': home_eff.get('defensiveRating', 100)
            }

        if not away_eff:
            away_rank = game.get('away_rank', 50)
            away_eff = {
                'adj_off': 110 - (away_rank / 10),
                'adj_def': 110 - (away_rank / 10)
            }
        else:
            away_eff = {
                'adj_off': away_eff.get('offensiveRating', 100),
                'adj_def': away_eff.get('defensiveRating', 100)
            }

        game_data = {
            'game_id': game.get('event_id'),
            'home_team': home_team,
            'away_team': away_team,
            'date': game.get('date'),
            'venue': game.get('venue'),
            'home_rank': game.get('home_rank'),
            'away_rank': game.get('away_rank'),
            'home_eff': home_eff,
            'away_eff': away_eff,
            'home_stats': stats_lookup.get(home_clean),
            'away_stats': stats_lookup.get(away_clean)
        }

        return game_data

    except Exception as e:
        print(f"Error fetching data for {game.get('home_team')} vs {game.get('away_team')}: {e}")
        return None

def generate_predictions():
    """Main function to generate predictions for all upcoming games."""
    print("Generating predictions for upcoming games...")

    # Load models
    models = load_models()
    if not models:
        print("No models found. Please run model training first.")
        return

    # Load team data once for all games
    efficiency_lookup, stats_lookup = load_team_data_once()

    # Load upcoming games
    upcoming_games = load_upcoming_games()
    if not upcoming_games:
        print("No upcoming games found.")
        return

    all_predictions = []
    successful_predictions = 0

    for game in upcoming_games:
        print(f"Predicting: {game['away_team']} @ {game['home_team']}")

        # Fetch game data using pre-loaded team data
        game_data = fetch_game_data(game, efficiency_lookup, stats_lookup)
        if not game_data:
            print(f"  Could not fetch data for this game")
            continue

        # Make predictions
        predictions = make_predictions(game_data, models)
        if not predictions:
            print(f"  Could not generate predictions for this game")
            continue

        # Combine game data with predictions
        game_result = {
            'game_info': game_data,
            'predictions': predictions,
            'generated_at': datetime.now().isoformat(),
            'season': 2025
        }

        all_predictions.append(game_result)
        successful_predictions += 1

        print(f"  Generated predictions")

    # Save predictions
    output_file = DATA_DIR / "upcoming_game_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2, default=str)

    print(f"\nSaved predictions for {successful_predictions}/{len(upcoming_games)} games to {output_file}")

    # Summary
    print("\nPrediction Summary:")
    print(f"  Total games: {len(upcoming_games)}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Models used: {len([k for k in models.keys() if not k.endswith('_scalers')])}")

if __name__ == "__main__":
    generate_predictions()