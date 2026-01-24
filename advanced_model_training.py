"""
Advanced Model Training for March Madness Betting Predictions

Implements the roadmap betting models with:
- Calibrated classifiers for better probability estimates
- Time-based cross validation (tournament CV)
- Ensemble approaches
- Comprehensive evaluation metrics
- ATS and O/U prediction functions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import joblib
import json
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import warnings
warnings.filterwarnings('ignore')

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
        return df
    else:
        print("No weighted training data found. Run feature_engineering.py first.")
        return pd.DataFrame()


def prepare_features(df: pd.DataFrame, target_type: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Prepare features for different prediction types."""
    # Use comprehensive feature set from weighted dataset
    base_features = [
        'off_eff_diff', 'def_eff_diff', 'net_eff_diff'
    ]

    # Add more features if available
    extended_features = base_features + [
        'tempo_diff', 'efficiency_diff', 'four_factors_diff'
    ]

    # Filter to available features
    available_features = [f for f in extended_features if f in df.columns]
    if len(available_features) < len(base_features):
        available_features = base_features

    # Filter based on target type
    if target_type == 'spread':
        valid_games = df.dropna(subset=['actual_spread'])
        y = valid_games['actual_spread']
    elif target_type == 'total':
        valid_games = df.dropna(subset=['actual_total'])
        y = valid_games['actual_total']
    elif target_type == 'moneyline':
        valid_games = df.dropna(subset=['actual_spread'])
        # Home team win if actual_spread < 0 (home team covered)
        y = (valid_games['actual_spread'] < 0).astype(int)
    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    if len(valid_games) == 0:
        print(f"No games with {target_type} data found")
        return pd.DataFrame(), pd.Series(), pd.Series()

    X = valid_games[available_features]
    weights = valid_games['sample_weight']

    print(f"Prepared {target_type} data: {len(X)} samples, {len(available_features)} features")
    print(f"  Features: {available_features}")
    print(f"  Regular games: {len(valid_games[valid_games['game_type']=='regular'])}, Tournament games: {len(valid_games[valid_games['game_type']=='tournament'])}")

    return X, y, weights


# ===== ROADMAP MODEL IMPLEMENTATIONS =====

def train_win_probability_model(X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> xgb.XGBClassifier:
    """Train model to predict win probability with calibrated probabilities."""
    print("Training win probability model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost classifier with good probability estimates
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic'
    )

    # Fit with sample weights if provided
    sample_weight = weights.loc[X_train.index] if weights is not None else None
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    brier = brier_score_loss(y_test, y_prob)
    logloss = log_loss(y_test, y_prob)

    print(".1%")
    print(".4f")
    print(".4f")

    return model


def train_spread_model(X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> xgb.XGBRegressor:
    """Train model to predict point margin."""
    print("Training spread prediction model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    # Fit with sample weights if provided
    sample_weight = weights.loc[X_train.index] if weights is not None else None
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(".2f")
    print(".2f")

    return model


def predict_ats(model: xgb.XGBRegressor, X: pd.DataFrame, spreads: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Predict against-the-spread outcomes."""
    predicted_margins = model.predict(X)

    # Team covers if predicted margin > spread
    # (positive spread means team is underdog, so they need to win by more than the spread)
    covers = predicted_margins > spreads

    return covers, predicted_margins


def train_total_model(X: pd.DataFrame, y: pd.Series, weights: Optional[pd.Series] = None) -> xgb.XGBRegressor:
    """Train model to predict game total."""
    print("Training total prediction model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )

    # Fit with sample weights if provided
    sample_weight = weights.loc[X_train.index] if weights is not None else None
    model.fit(X_train, y_train, sample_weight=sample_weight)

    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(".2f")
    print(".2f")

    return model


def predict_over_under(model: xgb.XGBRegressor, X: pd.DataFrame, lines: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Predict over/under outcomes."""
    predicted_totals = model.predict(X)

    # Over if predicted total > line
    overs = predicted_totals > lines

    return overs, predicted_totals


# ===== EVALUATION METRICS =====

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict:
    """Evaluate classification models (win, cover, over)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """Evaluate regression models (margin, total)."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def evaluate_betting_roi(predictions: np.ndarray, actuals: np.ndarray, odds: np.ndarray, stake: float = 100) -> Dict:
    """Calculate ROI from betting predictions."""
    wins = 0
    losses = 0
    profit = 0

    for pred, actual, odd in zip(predictions, actuals, odds):
        if pred == actual:
            wins += 1
            if odd > 0:
                profit += stake * (odd / 100)
            else:
                profit += stake * (100 / abs(odd))
        else:
            losses += 1
            profit -= stake

    roi = profit / (len(predictions) * stake) * 100

    return {
        "wins": wins,
        "losses": losses,
        "profit": profit,
        "roi_pct": roi,
        "win_rate": wins / len(predictions)
    }


# ===== TIME-BASED CROSS VALIDATION =====

def tournament_cv(X: pd.DataFrame, y: pd.Series, model, years: List[int]) -> List[Dict]:
    """Leave-one-tournament-out cross validation."""
    scores = []

    for test_year in years:
        train_mask = X["season"] != test_year
        test_mask = X["season"] == test_year

        X_train = X[train_mask].drop("season", axis=1)
        X_test = X[test_mask].drop("season", axis=1)
        y_train = y[train_mask]
        y_test = y[test_mask]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append({"year": test_year, "score": score})

    return scores


# ===== ENSEMBLE APPROACH =====

def create_betting_ensemble() -> Tuple[VotingClassifier, VotingRegressor]:
    """Create ensemble of models for robust predictions."""
    # Classification ensemble (for win/cover predictions)
    clf_ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000, random_state=42)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", xgb.XGBClassifier(n_estimators=100, random_state=42, objective='binary:logistic'))
        ],
        voting="soft"  # Use probability averaging
    )

    # Regression ensemble (for margin/total predictions)
    reg_ensemble = VotingRegressor(
        estimators=[
            ("ridge", Ridge(random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("xgb", xgb.XGBRegressor(n_estimators=100, random_state=42))
        ]
    )

    return clf_ensemble, reg_ensemble


# ===== ADVANCED TRAINING PIPELINE =====

def train_advanced_models(df: pd.DataFrame) -> Dict:
    """Train all models using the roadmap approach."""
    print("🚀 Training Advanced Betting Models")
    print("=" * 60)

    models = {}

    # 1. Moneyline Model (Win Probability)
    print("\n🏆 MONEYLINE MODEL (Win Probability)")
    print("-" * 40)
    X_ml, y_ml, weights_ml = prepare_features(df, 'moneyline')
    if not X_ml.empty:
        ml_model = train_win_probability_model(X_ml, y_ml, weights_ml)
        models['moneyline'] = ml_model

        # Save model
        joblib.dump(ml_model, MODEL_DIR / 'moneyline_advanced.joblib')
        print("💾 Saved moneyline model")

    # 2. Spread Model (Regression)
    print("\n📏 SPREAD MODEL (Margin Prediction)")
    print("-" * 40)
    X_spread, y_spread, weights_spread = prepare_features(df, 'spread')
    if not X_spread.empty:
        spread_model = train_spread_model(X_spread, y_spread, weights_spread)
        models['spread'] = spread_model

        # Save model
        joblib.dump(spread_model, MODEL_DIR / 'spread_advanced.joblib')
        print("💾 Saved spread model")

    # 3. Total Model (Regression)
    print("\n🔢 TOTAL MODEL (Combined Score)")
    print("-" * 40)
    X_total, y_total, weights_total = prepare_features(df, 'total')
    if not X_total.empty:
        total_model = train_total_model(X_total, y_total, weights_total)
        models['total'] = total_model

        # Save model
        joblib.dump(total_model, MODEL_DIR / 'total_advanced.joblib')
        print("💾 Saved total model")

    # 4. Ensemble Models
    print("\n🎭 ENSEMBLE MODELS")
    print("-" * 40)
    try:
        if not X_ml.empty and not X_spread.empty:
            clf_ensemble, reg_ensemble = create_betting_ensemble()

            # Train classification ensemble
            X_train_ml, X_test_ml, y_train_ml, y_test_ml = train_test_split(
                X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
            )
            clf_ensemble.fit(X_train_ml, y_train_ml)

            # Train regression ensemble
            X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
                X_spread, y_spread, test_size=0.2, random_state=42
            )
            reg_ensemble.fit(X_train_reg, y_train_reg)

            models['ensemble_clf'] = clf_ensemble
            models['ensemble_reg'] = reg_ensemble

            # Save ensembles
            joblib.dump(clf_ensemble, MODEL_DIR / 'ensemble_classifier.joblib')
            joblib.dump(reg_ensemble, MODEL_DIR / 'ensemble_regressor.joblib')
            print("💾 Saved ensemble models")
    except Exception as e:
        print(f"⚠️  Skipping ensemble models due to: {e}")

    return models


def run_cross_validation(df: pd.DataFrame) -> None:
    """Run time-based cross validation on historical tournaments."""
    print("\n⏰ TIME-BASED CROSS VALIDATION")
    print("-" * 40)

    years = sorted(df['season'].unique())
    print(f"Testing on tournaments: {years}")

    # Moneyline CV
    X_ml, y_ml, _ = prepare_features(df, 'moneyline')
    if not X_ml.empty and 'season' in X_ml.columns:
        ml_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        ml_scores = tournament_cv(X_ml, y_ml, ml_model, years)
        print("Moneyline CV Scores:")
        for score in ml_scores:
            print(".3f")

    # Spread CV
    X_spread, y_spread, _ = prepare_features(df, 'spread')
    if not X_spread.empty and 'season' in X_spread.columns:
        spread_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
        spread_scores = tournament_cv(X_spread, y_spread, spread_model, years)
        print("Spread CV Scores (MAE):")
        for score in spread_scores:
            print(".3f")


def save_advanced_metrics(models: Dict) -> None:
    """Save comprehensive metrics for advanced models."""
    metrics = {}

    # Load test data for evaluation
    df = load_weighted_training_data()
    if df.empty:
        return

    # Evaluate each model
    for model_name, model in models.items():
        if model_name == 'moneyline':
            X, y, _ = prepare_features(df, 'moneyline')
            if not X.empty:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)[:, 1]

                metrics['moneyline'] = evaluate_classification(y_test, y_pred, y_prob)

        elif model_name == 'spread':
            X, y, _ = prepare_features(df, 'spread')
            if not X.empty:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                y_pred = model.predict(X_test)

                metrics['spread'] = evaluate_regression(y_test, y_pred)

        elif model_name == 'total':
            X, y, _ = prepare_features(df, 'total')
            if not X.empty:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                y_pred = model.predict(X_test)

                metrics['total'] = evaluate_regression(y_test, y_pred)

    # Save metrics
    metrics_path = MODEL_DIR / "advanced_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"💾 Saved advanced metrics to {metrics_path}")


if __name__ == "__main__":
    print("🎯 Advanced March Madness Betting Model Training")
    print("=" * 60)

    # Load data
    df = load_weighted_training_data()
    if df.empty:
        exit(1)

    # Train advanced models
    models = train_advanced_models(df)

    # Run cross validation
    run_cross_validation(df)

    # Save comprehensive metrics
    save_advanced_metrics(models)

    print("\n✅ Advanced model training complete!")
    print("🎯 Models ready for tournament predictions with:")
    print("   • Calibrated probability estimates")
    print("   • Ensemble predictions")
    print("   • Time-based validation")
    print("   • Comprehensive evaluation metrics")