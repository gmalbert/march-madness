# Betting Models Implementation
# Based on roadmap-betting-models.md

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error
)
from xgboost import XGBClassifier, XGBRegressor
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# MODEL TRAINING FUNCTIONS
# =============================================================================

def train_win_probability_model(X: pd.DataFrame, y: pd.Series) -> CalibratedClassifierCV:
    """Train model to predict win probability.

    Args:
        X: Feature matrix
        y: Binary target (1 for home win, 0 for away win)

    Returns:
        Calibrated classifier for probability predictions
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Use LogisticRegression as base for calibration (more stable than XGBoost)
    base_model = LogisticRegression(max_iter=1000, random_state=42)

    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)

    return model


def train_spread_model(X: pd.DataFrame, y_margin: pd.Series) -> XGBRegressor:
    """Train model to predict point margin.

    Args:
        X: Feature matrix
        y_margin: Target margin (home_score - away_score)

    Returns:
        Trained regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_margin, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model


def train_total_model(X: pd.DataFrame, y_total: pd.Series) -> XGBRegressor:
    """Train model to predict game total.

    Args:
        X: Feature matrix
        y_total: Target total points (home_score + away_score)

    Returns:
        Trained regression model
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_total, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    return model


# =============================================================================
# PREDICTION FUNCTIONS
# =============================================================================

def predict_ats(model, X: pd.DataFrame, spreads: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Predict against-the-spread outcomes.

    Args:
        model: Trained spread model
        X: Feature matrix
        spreads: Betting spreads (positive = home underdog)

    Returns:
        Tuple of (covers, predicted_margins)
        covers: Boolean array where True means team covers
        predicted_margins: Predicted point margins
    """
    predicted_margins = model.predict(X)

    # Team covers if predicted margin > spread
    # (positive spread means team is underdog, so they need to win by more)
    covers = predicted_margins > (-spreads)

    return covers, predicted_margins


def predict_over_under(model, X: pd.DataFrame, lines: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    """Predict over/under outcomes.

    Args:
        model: Trained total model
        X: Feature matrix
        lines: Over/under lines

    Returns:
        Tuple of (overs, predicted_totals)
        overs: Boolean array where True means over hits
        predicted_totals: Predicted total points
    """
    predicted_totals = model.predict(X)

    overs = predicted_totals > lines

    return overs, predicted_totals


# =============================================================================
# EVALUATION FUNCTIONS
# =============================================================================

def evaluate_classification(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Evaluate classification models (win, cover, over).

    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels
        y_prob: Predicted probabilities

    Returns:
        Dictionary of evaluation metrics
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Evaluate regression models (margin, total).

    Args:
        y_true: True values
        y_pred: Predicted values

    Returns:
        Dictionary of evaluation metrics
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def evaluate_betting_roi(predictions: np.ndarray, actuals: np.ndarray,
                         odds: np.ndarray, stake: float = 100) -> Dict[str, Any]:
    """Calculate ROI from betting predictions.

    Args:
        predictions: Predicted outcomes (True/False)
        actuals: Actual outcomes (True/False)
        odds: American odds for each bet
        stake: Amount wagered per bet

    Returns:
        Dictionary with ROI statistics
    """
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

    total_bets = len(predictions)
    roi = profit / (total_bets * stake) * 100

    return {
        "wins": wins,
        "losses": losses,
        "total_bets": total_bets,
        "profit": profit,
        "roi_pct": roi,
        "win_rate": wins / total_bets if total_bets > 0 else 0
    }


def evaluate_betting_roi_from_df(df: pd.DataFrame, model_type: str = "moneyline") -> Dict[str, Any]:
    """Evaluate betting ROI from historical data DataFrame.

    Args:
        df: DataFrame with historical games and predictions
        model_type: Type of bet to evaluate ("moneyline", "spread", "total")

    Returns:
        Dictionary with comprehensive evaluation results
    """
    if df.empty:
        return {}

    results = {
        "overall_roi": 0,
        "total_bets": 0,
        "accuracy": 0,
        "brier_score": 0,
        "model_results": {}
    }

    # Define model types and their evaluation logic
    model_configs = {
        "moneyline": {
            "pred_col": "pred_home_win_prob",
            "actual_col": "home_win",
            "odds_col": "home_moneyline",
            "alt_odds_col": "away_moneyline"
        },
        "spread": {
            "pred_col": "pred_spread",
            "actual_col": "ats_result",  # 1 if home covers, 0 if away covers
            "odds_col": "home_spread_odds"
        },
        "total": {
            "pred_col": "pred_total",
            "actual_col": "over_result",  # 1 if over hits, 0 if under hits
            "odds_col": "over_odds"
        }
    }

    if model_type not in model_configs:
        return results

    config = model_configs[model_type]

    # Check if required columns exist
    required_cols = [config["pred_col"], config["actual_col"]]
    if not all(col in df.columns for col in required_cols):
        return results

    # Get valid data (non-null predictions and actuals)
    valid_mask = (
        df[config["pred_col"]].notna() &
        df[config["actual_col"]].notna()
    )

    if not valid_mask.any():
        return results

    df_valid = df[valid_mask].copy()

    # For moneyline, we need to determine which side to bet on
    if model_type == "moneyline":
        # Bet on home if predicted prob > 0.5, else bet on away
        bet_on_home = df_valid[config["pred_col"]] > 0.5
        actual_outcome = df_valid[config["actual_col"]].astype(bool)

        # Get appropriate odds
        odds = np.where(
            bet_on_home,
            df_valid[config["odds_col"]],
            df_valid[config["alt_odds_col"]]
        )

        # Predictions are correct if we bet on home and home won, or bet on away and away won
        predictions_correct = np.where(
            bet_on_home,
            actual_outcome,
            ~actual_outcome
        )

        # Probabilities for the side we bet on
        bet_probabilities = np.where(
            bet_on_home,
            df_valid[config["pred_col"]],
            1 - df_valid[config["pred_col"]]
        )

    else:
        # For spread and total, use direct predictions
        predictions_correct = (df_valid[config["pred_col"]] > 0.5).astype(bool)
        actual_outcome = df_valid[config["actual_col"]].astype(bool)
        odds = df_valid[config["odds_col"]]
        bet_probabilities = df_valid[config["pred_col"]]

    # Calculate ROI
    roi_results = evaluate_betting_roi(
        predictions_correct,
        actual_outcome.values,
        odds
    )

    # Calculate additional metrics
    accuracy = accuracy_score(actual_outcome.values, predictions_correct)
    brier_score = brier_score_loss(actual_outcome.values, bet_probabilities)

    # Calculate MAE and RMSE for regression-style predictions
    mae = mean_absolute_error(actual_outcome.values.astype(int), bet_probabilities)
    rmse = np.sqrt(mean_squared_error(actual_outcome.values.astype(int), bet_probabilities))

    results.update({
        "overall_roi": roi_results["roi_pct"],
        "total_bets": roi_results["total_bets"],
        "accuracy": accuracy,
        "brier_score": brier_score,
        "mae": mae,
        "rmse": rmse,
        "model_results": {
            "overall": {
                "roi": roi_results["roi_pct"],
                "accuracy": accuracy,
                "brier_score": brier_score,
                "mae": mae,
                "rmse": rmse,
                "total_bets": roi_results["total_bets"]
            }
        }
    })

    return results


def evaluate_model_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Evaluate model calibration - how well predicted probabilities match actual outcomes.

    Args:
        y_true: True binary labels
        y_prob: Predicted probabilities
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    # Create bins for predicted probabilities
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    # Calculate calibration curve
    observed_probs = []
    predicted_probs = []

    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if np.sum(mask) > 0:
            observed_prob = np.mean(y_true[mask])
            predicted_prob = np.mean(y_prob[mask])
            observed_probs.append(observed_prob)
            predicted_probs.append(predicted_prob)

    # Calculate calibration error (mean absolute difference)
    if observed_probs:
        calibration_error = np.mean(np.abs(np.array(observed_probs) - np.array(predicted_probs)))
    else:
        calibration_error = 0.0

    # Calculate Brier score
    brier_score = brier_score_loss(y_true, y_prob)

    return {
        "brier_score": brier_score,
        "calibration_error": calibration_error,
        "bin_centers": bin_centers.tolist(),
        "observed_probs": observed_probs,
        "predicted_probs": predicted_probs,
        "n_bins": n_bins
    }


# =============================================================================
# CROSS VALIDATION
# =============================================================================

def tournament_cv(X: pd.DataFrame, y: pd.Series, model: Any, years: List[int]) -> List[Dict]:
    """Leave-one-tournament-out cross validation.

    Args:
        X: Feature matrix with 'year' column
        y: Target variable
        model: Model to evaluate
        years: List of years to test on

    Returns:
        List of score dictionaries for each year
    """
    scores = []

    for test_year in years:
        train_mask = X["year"] != test_year
        test_mask = X["year"] == test_year

        X_train = X[train_mask].drop("year", axis=1)
        X_test = X[test_mask].drop("year", axis=1)
        y_train = y[train_mask]
        y_test = y[test_mask]

        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append({"year": test_year, "score": score})

    return scores


# =============================================================================
# ENSEMBLE MODELS
# =============================================================================

def create_betting_ensemble_from_df(df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
    """Create and evaluate ensemble model from historical data.

    Args:
        df: DataFrame with historical games and predictions
        model_type: Type of bet to evaluate ("moneyline", "spread", "total")

    Returns:
        Dictionary with ensemble evaluation results
    """
    if df.empty:
        return {}

    # For now, return a simple ensemble that averages predictions
    # In a real implementation, this would train actual ensemble models

    try:
        # Evaluate using the same logic as individual models
        results = evaluate_betting_roi_from_df(df, model_type)

        # Add some ensemble-specific logic (for now, just return the same results)
        # In practice, you'd train multiple models and combine their predictions

        return results.get('model_results', {}).get('overall', {})

    except Exception as e:
        print(f"Ensemble creation failed: {e}")
        return {}


def evaluate_model_calibration(df: pd.DataFrame, model_type: str, n_bins: int = 10) -> Dict[str, Any]:
    """Evaluate model calibration from historical data DataFrame.

    Args:
        df: DataFrame with historical games and predictions
        model_type: Type of bet to evaluate ("moneyline", "spread", "total")
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    if df.empty:
        return {}

    # Define columns based on model type
    config = {
        "moneyline": {
            "prob_col": "pred_home_win_prob",
            "actual_col": "home_win"
        },
        "spread": {
            "prob_col": "pred_spread_prob",  # Assuming this exists
            "actual_col": "ats_result"
        },
        "total": {
            "prob_col": "pred_total_prob",  # Assuming this exists
            "actual_col": "over_result"
        }
    }

    if model_type not in config:
        return {}

    prob_col = config[model_type]["prob_col"]
    actual_col = config[model_type]["actual_col"]

    # Check if columns exist
    if prob_col not in df.columns or actual_col not in df.columns:
        return {}

    # Get valid data
    valid_mask = (
        df[prob_col].notna() &
        df[actual_col].notna() &
        (df[prob_col] >= 0) &
        (df[prob_col] <= 1)
    )

    if not valid_mask.any():
        return {}

    y_true = df.loc[valid_mask, actual_col].values.astype(int)
    y_prob = df.loc[valid_mask, prob_col].values

    # Use the existing calibration function
    return evaluate_model_calibration(y_true, y_prob, n_bins)


# =============================================================================
# ENSEMBLE MODELS
# =============================================================================

def create_betting_ensemble():
    """Create ensemble of models for robust predictions.

    Returns:
        Tuple of (classification_ensemble, regression_ensemble)
    """
    # Classification ensemble (for win/cover predictions)
    clf_ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
            ("xgb", XGBClassifier(n_estimators=100, random_state=42))
        ],
        voting="soft"
    )

    # Regression ensemble (for margin/total predictions)
    reg_ensemble = VotingRegressor(
        estimators=[
            ("ridge", Ridge(random_state=42)),
            ("rf", RandomForestRegressor(n_estimators=100, random_state=42)),
            ("xgb", XGBRegressor(n_estimators=100, random_state=42))
        ]
    )

    return clf_ensemble, reg_ensemble


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_model_summary() -> pd.DataFrame:
    """Get summary of model types and metrics.

    Returns:
        DataFrame with model summary information
    """
    data = {
        "Bet Type": ["Moneyline", "Spread", "Over/Under", "Value"],
        "Model Type": ["Classification", "Regression → Classification", "Regression → Classification", "Probability Comparison"],
        "Target": ["Win (0/1)", "Margin → Cover", "Total → Over", "Edge"],
        "Key Metric": ["Brier Score", "ATS Accuracy", "O/U Accuracy", "ROI"]
    }

    return pd.DataFrame(data)


def validate_dependencies():
    """Validate that required dependencies are installed."""
    required_packages = [
        'pandas', 'numpy', 'sklearn', 'xgboost'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        raise ImportError(f"Missing required packages: {', '.join(missing)}")

    print("✅ All betting model dependencies are available")


if __name__ == "__main__":
    # Quick validation
    validate_dependencies()
    print("Betting models module loaded successfully!")
    print("\nModel Summary:")
    print(get_model_summary())