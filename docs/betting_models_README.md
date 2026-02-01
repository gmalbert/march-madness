# Betting Models Implementation

This module implements the betting model training and evaluation functions described in `docs/roadmap-betting-models.md`.

## Overview

The `betting_models.py` module provides comprehensive betting prediction capabilities for:

- **Moneyline**: Win probability classification
- **Spread**: Point margin regression → ATS classification
- **Over/Under**: Total points regression → O/U classification
- **Value Betting**: Probability comparison for edge detection

## Quick Start

```python
from betting_models import (
    train_win_probability_model,
    train_spread_model,
    train_total_model,
    evaluate_classification,
    evaluate_betting_roi
)

# Train models
moneyline_model = train_win_probability_model(X_train, y_win)
spread_model = train_spread_model(X_train, y_margin)
total_model = train_total_model(X_train, y_total)

# Make predictions
win_probs = moneyline_model.predict_proba(X_test)[:, 1]
covers, margins = predict_ats(spread_model, X_test, spreads)
overs, totals = predict_over_under(total_model, X_test, ou_lines)

# Evaluate performance
clf_metrics = evaluate_classification(y_test, win_preds, win_probs)
roi_metrics = evaluate_betting_roi(predictions, actuals, odds)
```

## Model Types

| Bet Type | Model Type | Target | Key Metric |
|----------|------------|--------|------------|
| Moneyline | Classification | Win (0/1) | Brier Score |
| Spread | Regression → Classification | Margin → Cover | ATS Accuracy |
| Over/Under | Regression → Classification | Total → Over | O/U Accuracy |
| Value | Probability Comparison | Edge | ROI |

## Functions

### Training Functions

- `train_win_probability_model(X, y)` - Train calibrated classifier for win probabilities
- `train_spread_model(X, y_margin)` - Train regression model for point margins
- `train_total_model(X, y_total)` - Train regression model for total points

### Prediction Functions

- `predict_ats(model, X, spreads)` - Predict against-the-spread outcomes
- `predict_over_under(model, X, lines)` - Predict over/under outcomes

### Evaluation Functions

- `evaluate_classification(y_true, y_pred, y_prob)` - Classification metrics (accuracy, brier, log_loss)
- `evaluate_regression(y_true, y_pred)` - Regression metrics (MAE, RMSE)
- `evaluate_betting_roi(predictions, actuals, odds, stake=100)` - ROI calculation

### Advanced Functions

- `tournament_cv(X, y, model, years)` - Time-based cross validation
- `create_betting_ensemble()` - Create voting ensemble models
- `get_model_summary()` - Get model type summary table

## Dependencies

All required packages are in `requirements.txt`:
- scikit-learn
- xgboost
- pandas
- numpy

## Testing

Run the comprehensive test suite:

```bash
python test_betting_models.py
```

This validates all functions work correctly with sample data.

## Integration

The betting models integrate with the existing March Madness prediction system:

1. **Feature Engineering**: Use `feature_engineering.py` to create input features
2. **Model Training**: Use functions from this module to train models
3. **Predictions**: Use trained models in `predictions.py` for real-time betting predictions
4. **Evaluation**: Use evaluation functions to track model performance

## Example Usage in Streamlit App

```python
# In predictions.py
from betting_models import train_win_probability_model, evaluate_betting_roi

# Train model
model = train_win_probability_model(X_train, y_train)

# Use in predictions
win_prob = model.predict_proba(game_features)[0][1]

# Calculate betting value
implied_prob = moneyline_to_implied_probability(moneyline_odds)
edge = win_prob - implied_prob
```

## Model Persistence

Models can be saved/loaded using joblib:

```python
import joblib

# Save
joblib.dump(model, 'models/moneyline_model.joblib')

# Load
model = joblib.load('models/moneyline_model.joblib')
```

## Performance Metrics

Track these key metrics for model evaluation:

- **Classification**: Brier Score (lower is better), Accuracy
- **Regression**: MAE/RMSE (lower is better)
- **Betting**: ROI % (higher is better), Win Rate

## Next Steps

1. Integrate with existing feature engineering pipeline
2. Add hyperparameter tuning
3. Implement model versioning and A/B testing
4. Add confidence intervals for predictions
5. Create automated retraining pipeline</content>
<parameter name="filePath">c:\Users\gmalb\Downloads\march-madness\BETTING_MODELS_README.md