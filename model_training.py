# Model Training for March Madness Betting Predictions

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import os
from pathlib import Path
from typing import Dict, Tuple, Optional

# Data directory
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

def load_weighted_training_data() -> pd.DataFrame:
    """Load weighted training data with sample weights."""
    data_path = DATA_DIR / "training_data_weighted.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} weighted games from 2016-2025")
        print(f"Years covered: {sorted(df['season'].unique())}")
        print(f"Game types: {df['game_type'].value_counts().to_dict()}")
        print(f"Sample weights - Regular: {df[df['game_type']=='regular']['sample_weight'].iloc[0] if len(df[df['game_type']=='regular']) > 0 else 'N/A'}, Tournament: {df[df['game_type']=='tournament']['sample_weight'].iloc[0] if len(df[df['game_type']=='tournament']) > 0 else 'N/A'}")
        return df
    else:
        print("No weighted training data found. Run feature_engineering.py first.")
        return pd.DataFrame()

def prepare_spread_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for spread prediction."""
    # Use basic efficiency features available in weighted dataset
    spread_features = [
        'off_eff_diff', 'def_eff_diff', 'net_eff_diff'
    ]

    # Filter to games with actual spread results
    valid_games = df.dropna(subset=['actual_spread'])

    if len(valid_games) == 0:
        print("No games with actual spread data found")
        return pd.DataFrame(), pd.Series(), pd.Series()

    X = valid_games[spread_features]
    y = valid_games['actual_spread']
    weights = valid_games['sample_weight']

    print(f"Prepared spread data: {len(X)} samples, {len(spread_features)} features")
    print(f"  Regular games: {len(valid_games[valid_games['game_type']=='regular'])}, Tournament games: {len(valid_games[valid_games['game_type']=='tournament'])}")
    return X, y, weights

def prepare_total_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for total prediction."""
    # Use basic efficiency features for total prediction
    total_features = [
        'off_eff_diff', 'def_eff_diff', 'net_eff_diff'
    ]

    # Filter to games with actual total results
    valid_games = df.dropna(subset=['actual_total'])

    if len(valid_games) == 0:
        print("No games with actual total data found")
        return pd.DataFrame(), pd.Series(), pd.Series()

    X = valid_games[total_features]
    y = valid_games['actual_total']
    weights = valid_games['sample_weight']

    print(f"Prepared total data: {len(X)} samples, {len(total_features)} features")
    print(f"  Regular games: {len(valid_games[valid_games['game_type']=='regular'])}, Tournament games: {len(valid_games[valid_games['game_type']=='tournament'])}")
    return X, y, weights

def prepare_moneyline_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare data for moneyline prediction (classification)."""
    # Use same basic features as spread but predict win/loss
    spread_features = [
        'off_eff_diff', 'def_eff_diff', 'net_eff_diff'
    ]

    # Filter to games with actual results
    valid_games = df.dropna(subset=['actual_spread'])

    if len(valid_games) == 0:
        print("No games with moneyline data found")
        return pd.DataFrame(), pd.Series(), pd.Series()

    X = valid_games[spread_features]
    # Home team win if actual_spread > 0 (home team covered the spread)
    y = (valid_games['actual_spread'] < 0).astype(int)  # 1 if home team won, 0 if away team won
    weights = valid_games['sample_weight']

    print(f"Prepared moneyline data: {len(X)} samples, {len(spread_features)} features")
    print(f"  Regular games: {len(valid_games[valid_games['game_type']=='regular'])}, Tournament games: {len(valid_games[valid_games['game_type']=='tournament'])}")
    return X, y, weights

def train_spread_model(X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> Dict:
    """Train models for spread prediction."""
    print("Training spread prediction models...")

    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train, sample_weight=weights_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    models['linear_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'mae': lr_mae,
        'rmse': lr_rmse,
        'predictions': lr_pred,
        'actual': y_test
    }

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train, sample_weight=weights_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    models['random_forest'] = {
        'model': rf_model,
        'scaler': None,
        'mae': rf_mae,
        'rmse': rf_rmse,
        'predictions': rf_pred,
        'actual': y_test
    }

    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train, sample_weight=weights_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    models['xgboost'] = {
        'model': xgb_model,
        'scaler': None,
        'mae': xgb_mae,
        'rmse': xgb_rmse,
        'predictions': xgb_pred,
        'actual': y_test
    }

    print("Spread model results:")
    print(".2f")
    print(".2f")
    print(".2f")

    return models

def train_total_model(X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> Dict:
    """Train models for total prediction."""
    print("Training total prediction models...")

    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train, sample_weight=weights_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))

    models['linear_regression'] = {
        'model': lr_model,
        'scaler': scaler,
        'mae': lr_mae,
        'rmse': lr_rmse,
        'predictions': lr_pred,
        'actual': y_test
    }

    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train, sample_weight=weights_train)
    rf_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))

    models['random_forest'] = {
        'model': rf_model,
        'scaler': None,
        'mae': rf_mae,
        'rmse': rf_rmse,
        'predictions': rf_pred,
        'actual': y_test
    }

    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train, sample_weight=weights_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))

    models['xgboost'] = {
        'model': xgb_model,
        'scaler': None,
        'mae': xgb_mae,
        'rmse': xgb_rmse,
        'predictions': xgb_pred,
        'actual': y_test
    }

    print("Total model results:")
    print(".2f")
    print(".2f")
    print(".2f")

    return models

def train_moneyline_model(X: pd.DataFrame, y: pd.Series, weights: pd.Series) -> Dict:
    """Train models for moneyline prediction."""
    print("Training moneyline prediction models...")

    # Split data
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(X, y, weights, test_size=0.2, random_state=42, stratify=y)

    models = {}

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train, sample_weight=weights_train)
    lr_pred = lr_model.predict(X_test)
    lr_accuracy = accuracy_score(y_test, lr_pred)

    models['logistic_regression'] = {
        'model': lr_model,
        'accuracy': lr_accuracy,
        'predictions': lr_pred,
        'actual': y_test
    }

    # Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train, sample_weight=weights_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    models['random_forest'] = {
        'model': rf_model,
        'accuracy': rf_accuracy,
        'predictions': rf_pred,
        'actual': y_test
    }

    # XGBoost
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train, sample_weight=weights_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_pred)

    models['xgboost'] = {
        'model': xgb_model,
        'accuracy': xgb_accuracy,
        'predictions': xgb_pred,
        'actual': y_test
    }

    print("Moneyline model results:")
    print(".1%")
    print(".1%")
    print(".1%")

    return models

def save_models(models: Dict, model_type: str):
    """Save trained models to disk."""
    for model_name, model_data in models.items():
        model_path = MODEL_DIR / f"{model_type}_{model_name}.joblib"
        joblib.dump(model_data['model'], model_path)

        if model_data.get('scaler'):
            scaler_path = MODEL_DIR / f"{model_type}_{model_name}_scaler.joblib"
            joblib.dump(model_data['scaler'], scaler_path)

        print(f"Saved {model_type} {model_name} model to {model_path}")

def save_model_metrics(models: Dict, model_type: str):
    """Save model performance metrics to a JSON file."""
    metrics = {}
    
    if model_type in ['spread', 'total']:
        # For regression models, calculate ensemble MAE
        mae_values = []
        for model_name, model_data in models.items():
            if 'mae' in model_data:
                mae_values.append(model_data['mae'])
        
        if mae_values:
            metrics['mae'] = np.mean(mae_values)
            metrics['mae_range'] = f"{min(mae_values):.2f} - {max(mae_values):.2f}"
    
    elif model_type == 'moneyline':
        # For classification models, calculate ensemble accuracy
        accuracy_values = []
        for model_name, model_data in models.items():
            if 'accuracy' in model_data:
                accuracy_values.append(model_data['accuracy'])
        
        if accuracy_values:
            metrics['accuracy'] = np.mean(accuracy_values)
            metrics['accuracy_range'] = f"{min(accuracy_values):.1%} - {max(accuracy_values):.1%}"
    
    if metrics:
        metrics_path = MODEL_DIR / f"{model_type}_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved {model_type} metrics to {metrics_path}")

if __name__ == "__main__":
    print("🏀 Training March Madness betting prediction models on tournament data...")

    # Load weighted training data (regular season + tournament with weights)
    df = load_weighted_training_data()

    if df.empty:
        print("No comprehensive training data available. Run feature_engineering.py first.")
        exit(1)

    print("\n📊 Dataset Overview:")
    print(f"   Total games: {len(df)}")
    print(f"   Years: {sorted(df['season'].unique())}")

    # Check for betting data columns
    betting_cols = [c for c in df.columns if 'betting' in c.lower() or 'moneyline' in c.lower()]
    if betting_cols:
        print(f"   Games with betting lines: {df[betting_cols[0]].notna().sum()}")
    else:
        print("   Games with betting lines: 0 (tournament games)")

    print(f"   Games with results: {df['actual_spread'].notna().sum()}")

    # Train spread models
    print("\n" + "="*60)
    print("SPREAD PREDICTION MODELS (TOURNAMENT 2016-2025)")
    print("="*60)
    X_spread, y_spread, weights_spread = prepare_spread_data(df)
    if not X_spread.empty:
        spread_models = train_spread_model(X_spread, y_spread, weights_spread)
        save_models(spread_models, 'spread')
        save_model_metrics(spread_models, 'spread')

    # Train total models
    print("\n" + "="*60)
    print("TOTAL PREDICTION MODELS (TOURNAMENT 2016-2025)")
    print("="*60)
    X_total, y_total, weights_total = prepare_total_data(df)
    if not X_total.empty:
        total_models = train_total_model(X_total, y_total, weights_total)
        save_models(total_models, 'total')
        save_model_metrics(total_models, 'total')

    # Train moneyline models
    print("\n" + "="*60)
    print("MONEYLINE PREDICTION MODELS (TOURNAMENT 2016-2025)")
    print("="*60)
    X_moneyline, y_moneyline, weights_moneyline = prepare_moneyline_data(df)
    if not X_moneyline.empty:
        moneyline_models = train_moneyline_model(X_moneyline, y_moneyline, weights_moneyline)
        save_models(moneyline_models, 'moneyline')
        save_model_metrics(moneyline_models, 'moneyline')

    print("\n✅ Model retraining complete!")
    print(f"Models saved to {MODEL_DIR}")
    print("🎯 Ready for improved betting predictions trained on actual tournament games!")