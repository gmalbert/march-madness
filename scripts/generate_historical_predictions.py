#!/usr/bin/env python3
"""
Generate predictions for historical betting data using trained models.

This script loads the historical games with betting data and uses the trained
models to generate predictions for evaluation purposes.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from predictions import calculate_features
from predictions import normalize_team_name

# Data directory
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"

def load_trained_models():
    """Load all trained models."""
    models = {}

    # Load scalers
    try:
        models['spread_scaler'] = joblib.load(MODEL_DIR / 'spread_linear_regression_scaler.joblib')
    except:
        models['spread_scaler'] = None

    try:
        models['total_scaler'] = joblib.load(MODEL_DIR / 'total_linear_regression_scaler.joblib')
    except:
        models['total_scaler'] = None

    # Load spread models
    models['spread'] = {}
    for model_name in ['xgboost', 'random_forest', 'linear']:
        try:
            models['spread'][model_name] = joblib.load(MODEL_DIR / f'spread_{model_name}.joblib')
        except:
            print(f"Warning: Could not load spread_{model_name}.joblib")
            models['spread'][model_name] = None

    # Load total models
    models['total'] = {}
    for model_name in ['xgboost', 'random_forest', 'linear']:
        try:
            models['total'][model_name] = joblib.load(MODEL_DIR / f'total_{model_name}.joblib')
        except:
            print(f"Warning: Could not load total_{model_name}.joblib")
            models['total'][model_name] = None

    # Load moneyline models
    models['moneyline'] = {}
    for model_name in ['xgboost', 'random_forest', 'logistic']:
        try:
            models['moneyline'][model_name] = joblib.load(MODEL_DIR / f'moneyline_{model_name}.joblib')
        except:
            print(f"Warning: Could not load moneyline_{model_name}.joblib")
            models['moneyline'][model_name] = None

    return models

def generate_predictions_for_historical_data():
    """Generate predictions for historical betting data."""

    print("Loading historical betting data...")
    historical_file = DATA_DIR / "historical_games_with_betting.csv"

    if not historical_file.exists():
        print("Error: historical_games_with_betting.csv not found")
        return

    df = pd.read_csv(historical_file)
    print(f"Loaded {len(df)} historical games")

    # Load trained models
    print("Loading trained models...")
    models = load_trained_models()

    # Load team stats for feature calculation
    print("Loading team efficiency data...")
    try:
        from predictions import get_team_data
        efficiency_list, stats_list, season_used = get_team_data(2023)  # Use 2023 data
        print(f"Using {season_used} season data for {len(efficiency_list)} teams")
    except Exception as e:
        print(f"Error loading team data: {e}")
        return

    # Create lookup dictionaries
    efficiency_lookup = {normalize_team_name(team['team']): team for team in efficiency_list}
    stats_lookup = {normalize_team_name(team['team']): team for team in stats_list}

    # Initialize prediction columns
    df['pred_home_win_prob'] = 0.5
    df['pred_spread'] = 0.0
    df['pred_total'] = 0.0

    predictions_made = 0

    print("Generating predictions for each game...")

    for idx, row in df.iterrows():
        try:
            # Get team data
            home_team = normalize_team_name(row['home_team'])
            away_team = normalize_team_name(row['away_team'])

            # Skip if we don't have data for both teams
            if home_team not in efficiency_lookup or away_team not in efficiency_lookup:
                continue
            if home_team not in stats_lookup or away_team not in stats_lookup:
                continue

            # Create game data structure
            game_data = {
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'home_eff': efficiency_lookup[home_team],
                'away_eff': efficiency_lookup[away_team],
                'home_stats': stats_lookup[home_team],
                'away_stats': stats_lookup[away_team],
                'betting_spread': row.get('betting_spread'),
                'betting_over_under': row.get('betting_over_under'),
                'home_moneyline': row.get('home_moneyline'),
                'away_moneyline': row.get('away_moneyline')
            }

            # Calculate features
            features = calculate_features(
                game_data['home_stats'],
                game_data['away_stats'],
                game_data['home_eff'],
                game_data['away_eff']
            )

            # Generate predictions
            predictions = {'moneyline': {}, 'spread': {}, 'total': {}}

            # Moneyline predictions
            moneyline_probs = []
            if models['moneyline']:
                moneyline_df = pd.DataFrame([features['moneyline']], columns=[
                    'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
                    'kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff',
                    'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff',
                    'bart_oe_diff', 'bart_de_diff'
                ])

                for model_name, model in models['moneyline'].items():
                    if model is None:
                        continue
                    try:
                        if model_name == 'logistic':
                            scaler = models.get('moneyline_scalers', {}).get(model_name)
                            if scaler:
                                scaled_df = scaler.transform(moneyline_df)
                                prob = model.predict_proba(scaled_df)[0][1]
                            else:
                                prob = model.predict_proba(moneyline_df)[0][1]
                        else:
                            prob = model.predict_proba(moneyline_df)[0][1]
                        moneyline_probs.append(prob)
                    except:
                        continue

                if moneyline_probs:
                    df.at[idx, 'pred_home_win_prob'] = np.mean(moneyline_probs)

            # Spread predictions
            spread_preds = []
            if models['spread']:
                spread_df = pd.DataFrame([features['spread']], columns=[
                    'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
                    'kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff',
                    'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff',
                    'bart_oe_diff', 'bart_de_diff'
                ])

                for model_name, model in models['spread'].items():
                    if model is None:
                        continue
                    try:
                        if model_name == 'linear' and models['spread_scaler']:
                            scaled_df = models['spread_scaler'].transform(spread_df)
                            pred = model.predict(scaled_df)[0]
                        else:
                            pred = model.predict(spread_df)[0]
                        spread_preds.append(pred)
                    except:
                        continue

                if spread_preds:
                    df.at[idx, 'pred_spread'] = np.mean(spread_preds)

            # Total predictions
            total_preds = []
            if models['total']:
                total_df = pd.DataFrame([features['total']], columns=[
                    'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
                    'kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff',
                    'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff',
                    'bart_oe_diff', 'bart_de_diff'
                ])

                for model_name, model in models['total'].items():
                    if model is None:
                        continue
                    try:
                        if model_name == 'linear' and models['total_scaler']:
                            scaled_df = models['total_scaler'].transform(total_df)
                            pred = model.predict(scaled_df)[0]
                        else:
                            pred = model.predict(total_df)[0]
                        total_preds.append(pred)
                    except:
                        continue

                if total_preds:
                    df.at[idx, 'pred_total'] = np.mean(total_preds)

            predictions_made += 1

            if predictions_made % 100 == 0:
                print(f"Generated predictions for {predictions_made} games...")

        except Exception as e:
            # Skip games with errors
            continue

    print(f"Generated predictions for {predictions_made} out of {len(df)} games")

    # Save updated data
    output_file = DATA_DIR / "historical_games_with_betting_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

    # Show summary statistics
    print("\nPrediction Summary:")
    print(f"Games with moneyline predictions: {len(df[df['pred_home_win_prob'] != 0.5])}")
    print(f"Games with spread predictions: {len(df[df['pred_spread'] != 0.0])}")
    print(f"Games with total predictions: {len(df[df['pred_total'] != 0.0])}")

    print(".3f")
    print(".1f")
    print(".1f")

if __name__ == "__main__":
    generate_predictions_for_historical_data()