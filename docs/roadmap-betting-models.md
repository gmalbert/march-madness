# Roadmap: Betting Models

*Model approaches for spread, over/under, and moneyline predictions.*

## Model Types by Bet

### 1. Moneyline (Win Probability)
Binary classification with probability output.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def train_win_probability_model(X, y):
    """Train model to predict win probability."""
    from sklearn.model_selection import train_test_split
    from sklearn.calibration import CalibratedClassifierCV
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # XGBoost with calibration for better probabilities
    base_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    model = CalibratedClassifierCV(base_model, method="isotonic", cv=5)
    model.fit(X_train, y_train)
    
    return model
```

**✅ IMPLEMENTED** - Available in `betting_models.py` as `train_win_probability_model()` with LogisticRegression base model for better calibration

### 2. Spread Prediction (Margin Regression)
Regression to predict point margin, then compare to spread.

```python
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

def train_spread_model(X, y_margin):
    """Train model to predict point margin."""
    from sklearn.model_selection import train_test_split
    
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

def predict_ats(model, X, spreads):
    """Predict against-the-spread outcomes."""
    predicted_margins = model.predict(X)
    
    # Team covers if predicted margin > spread
    # (positive spread means team is underdog)
    covers = predicted_margins > (-spreads)
    
    return covers, predicted_margins
```

**✅ IMPLEMENTED** - Available in `betting_models.py` as `train_spread_model()` and `predict_ats()`

### 3. Over/Under Prediction (Total Regression)
Regression to predict total points.

```python
def train_total_model(X, y_total):
    """Train model to predict game total."""
    from sklearn.model_selection import train_test_split
    
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

def predict_over_under(model, X, lines):
    """Predict over/under outcomes."""
    predicted_totals = model.predict(X)
    
    overs = predicted_totals > lines
    
    return overs, predicted_totals
```

**✅ IMPLEMENTED** - Available in `betting_models.py` as `train_total_model()` and `predict_over_under()`

## Evaluation Metrics

```python
from sklearn.metrics import (
    accuracy_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error
)

def evaluate_classification(y_true, y_pred, y_prob):
    """Evaluate classification models (win, cover, over)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
    }

def evaluate_regression(y_true, y_pred):
    """Evaluate regression models (margin, total)."""
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
    }

def evaluate_betting_roi(predictions, actuals, odds, stake=100):
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
```

**✅ IMPLEMENTED** - Available in `betting_models.py` as `evaluate_classification()`, `evaluate_regression()`, and `evaluate_betting_roi()` with additional enhanced versions

## Time-Based Cross Validation

```python
def tournament_cv(X, y, model, years):
    """Leave-one-tournament-out cross validation."""
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
```

**✅ IMPLEMENTED** - Available in `betting_models.py` as `tournament_cv()` with enhanced error handling

## Ensemble Approach

```python
def create_betting_ensemble():
    """Create ensemble of models for robust predictions."""
    from sklearn.ensemble import VotingClassifier, VotingRegressor
    
    # Classification ensemble (for win/cover predictions)
    clf_ensemble = VotingClassifier(
        estimators=[
            ("lr", LogisticRegression(max_iter=1000)),
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("xgb", XGBClassifier(n_estimators=100))
        ],
        voting="soft"
    )
    
    # Regression ensemble (for margin/total predictions)
    reg_ensemble = VotingRegressor(
        estimators=[
            ("ridge", Ridge()),
            ("rf", RandomForestRegressor(n_estimators=100)),
            ("xgb", XGBRegressor(n_estimators=100))
        ]
    )
    
    return clf_ensemble, reg_ensemble
```

**✅ IMPLEMENTED** - Available in `betting_models.py` as `create_betting_ensemble()` with additional ensemble creation functions

## Model Summary

| Bet Type | Model Type | Target | Key Metric |
|----------|------------|--------|------------|
| Moneyline | Classification | Win (0/1) | Brier Score |
| Spread | Regression → Classification | Margin → Cover | ATS Accuracy |
| Over/Under | Regression → Classification | Total → Over | O/U Accuracy |
| Value | Probability Comparison | Edge | ROI |

**✅ IMPLEMENTED** - Available in `betting_models.py` as `get_model_summary()` function that returns this table as a DataFrame

## Recommended Dependencies

```
# Add to requirements.txt
scikit-learn
xgboost
pandas
numpy
```

**✅ IMPLEMENTED** - All dependencies are available and validated via `validate_dependencies()` function in `betting_models.py`

## Next Steps
- See `roadmap-betting-features.md` for feature engineering
- See `roadmap-ui.md` for displaying predictions

**✅ ALL BETTING MODELS IMPLEMENTED** - The complete betting model framework is available in `betting_models.py` with comprehensive testing and validation
