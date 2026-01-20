# Roadmap: Modeling

*Suggested approaches for prediction models.*

## Model Types

### 1. Logistic Regression (Baseline)
Simple, interpretable, good starting point.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_model(X, y):
    """Train a baseline logistic regression model."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
    print(classification_report(y_test, y_pred))
    
    return model
```

### 2. Random Forest
Handles non-linear relationships, feature importance.

```python
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X, y, n_estimators=100):
    """Train a random forest classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Feature importance
    importance = dict(zip(X.columns, model.feature_importances_))
    print("Top features:", sorted(importance.items(), key=lambda x: -x[1])[:5])
    
    return model
```

### 3. Gradient Boosting (XGBoost)
High performance, handles complex patterns.

```python
# Add xgboost to requirements.txt
from xgboost import XGBClassifier

def train_xgboost(X, y):
    """Train XGBoost classifier."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    return model, y_prob
```

## Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV

def calibrate_model(model, X, y):
    """Calibrate model probabilities."""
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=5)
    calibrated.fit(X, y)
    return calibrated
```

## Evaluation Metrics

```python
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

def evaluate_probabilities(y_true, y_prob):
    """Evaluate probability predictions."""
    return {
        "brier_score": brier_score_loss(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "auc_roc": roc_auc_score(y_true, y_prob)
    }
```

## Cross-Validation Strategy

```python
from sklearn.model_selection import TimeSeriesSplit

def time_based_cv(X, y, model, n_splits=5):
    """Use time-based cross-validation for sports data."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
    
    return scores
```

## Model Comparison Framework

```python
def compare_models(X, y):
    """Compare multiple models."""
    models = {
        "Logistic": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(n_estimators=100)
    }
    
    results = {}
    for name, model in models.items():
        scores = time_based_cv(X, y, model)
        results[name] = {
            "mean": sum(scores) / len(scores),
            "std": (sum((s - sum(scores)/len(scores))**2 for s in scores) / len(scores)) ** 0.5
        }
    
    return results
```

## Model Recommendations

| Model | Use Case | Pros | Cons |
|-------|----------|------|------|
| Logistic Regression | Baseline | Interpretable, fast | Limited complexity |
| Random Forest | General | Feature importance | Can overfit |
| XGBoost | Best accuracy | High performance | Tuning required |
| Neural Network | Complex patterns | Flexible | Needs more data |

## Next Steps
- See `roadmap-features.md` for feature ideas
- See `roadmap-ui.md` for displaying predictions
