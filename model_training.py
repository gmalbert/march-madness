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

def load_training_data(year: int) -> pd.DataFrame:
    """Load training data for a given year."""
    data_path = DATA_DIR / f"training_data_{year}.csv"
    if data_path.exists():
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} games from {year}")
        return df
    else:
        print(f"No training data found for {year}")
        return pd.DataFrame()

def prepare_spread_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for spread prediction."""
    # Features for spread prediction
    spread_features = [
        'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
        'spread_net_rating_diff', 'spread_off_rating_diff', 'spread_def_rating_diff',
        'spread_ppg_diff', 'spread_opp_ppg_diff', 'spread_margin_diff',
        'spread_efg_diff', 'spread_to_rate_diff', 'spread_orb_diff', 'spread_ft_rate_diff'
    ]

    # Filter to games with actual spread results
    valid_games = df.dropna(subset=['actual_spread'])

    if len(valid_games) == 0:
        print("No games with actual spread data found")
        return pd.DataFrame(), pd.Series()

    X = valid_games[spread_features]
    y = valid_games['actual_spread']

    print(f"Prepared spread data: {len(X)} samples, {len(spread_features)} features")
    return X, y

def prepare_total_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for total prediction."""
    # Features for total prediction
    total_features = [
        'total_combined_off_eff', 'total_combined_def_eff', 'total_avg_off_eff', 'total_avg_def_eff',
        'total_combined_tempo', 'total_avg_tempo', 'total_combined_ppg', 'total_combined_opp_ppg',
        'total_combined_fg_pct', 'total_combined_3pt_pct', 'total_projected_total'
    ]

    # Filter to games with actual total results
    valid_games = df.dropna(subset=['actual_total'])

    if len(valid_games) == 0:
        print("No games with actual total data found")
        return pd.DataFrame(), pd.Series()

    X = valid_games[total_features]
    y = valid_games['actual_total']

    print(f"Prepared total data: {len(X)} samples, {len(total_features)} features")
    return X, y

def prepare_moneyline_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare data for moneyline prediction (classification)."""
    # Use same features as spread but predict win/loss
    spread_features = [
        'off_eff_diff', 'def_eff_diff', 'net_eff_diff',
        'spread_net_rating_diff', 'spread_off_rating_diff', 'spread_def_rating_diff',
        'spread_ppg_diff', 'spread_opp_ppg_diff', 'spread_margin_diff',
        'spread_efg_diff', 'spread_to_rate_diff', 'spread_orb_diff', 'spread_ft_rate_diff'
    ]

    # Filter to games with actual results
    valid_games = df.dropna(subset=['actual_spread'])

    if len(valid_games) == 0:
        print("No games with moneyline data found")
        return pd.DataFrame(), pd.Series()

    X = valid_games[spread_features]
    # Home team win if actual_spread > 0 (home team covered the spread)
    y = (valid_games['actual_spread'] < 0).astype(int)  # 1 if home team won, 0 if away team won

    print(f"Prepared moneyline data: {len(X)} samples, {len(spread_features)} features")
    return X, y

def train_spread_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train models for spread prediction."""
    print("Training spread prediction models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
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
    rf_model.fit(X_train, y_train)
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
    xgb_model.fit(X_train, y_train)
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

def train_total_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train models for total prediction."""
    print("Training total prediction models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
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
    rf_model.fit(X_train, y_train)
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
    xgb_model.fit(X_train, y_train)
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

def train_moneyline_model(X: pd.DataFrame, y: pd.Series) -> Dict:
    """Train models for moneyline prediction."""
    print("Training moneyline prediction models...")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {}

    # Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train, y_train)
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
    rf_model.fit(X_train, y_train)
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
    xgb_model.fit(X_train, y_train)
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

if __name__ == "__main__":
    print("🏈 Training March Madness betting prediction models...")

    # Load training data
    df = load_training_data(2022)

    if df.empty:
        print("No training data available")
        exit(1)

    # Train spread models
    print("\n" + "="*50)
    print("SPREAD PREDICTION MODELS")
    print("="*50)
    X_spread, y_spread = prepare_spread_data(df)
    if not X_spread.empty:
        spread_models = train_spread_model(X_spread, y_spread)
        save_models(spread_models, 'spread')

    # Train total models
    print("\n" + "="*50)
    print("TOTAL PREDICTION MODELS")
    print("="*50)
    X_total, y_total = prepare_total_data(df)
    if not X_total.empty:
        total_models = train_total_model(X_total, y_total)
        save_models(total_models, 'total')

    # Train moneyline models
    print("\n" + "="*50)
    print("MONEYLINE PREDICTION MODELS")
    print("="*50)
    X_moneyline, y_moneyline = prepare_moneyline_data(df)
    if not X_moneyline.empty:
        moneyline_models = train_moneyline_model(X_moneyline, y_moneyline)
        save_models(moneyline_models, 'moneyline')

    print("\n✅ Model training complete!")
    print(f"Models saved to {MODEL_DIR}")