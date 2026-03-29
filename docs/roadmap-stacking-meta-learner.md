# Roadmap: Stacking Meta-Learner Ensemble

> **Priority**: P2 — Meaningful improvement once base models are solid  
> **Estimated Effort**: Medium  
> **Source**: sports-quant `_meta_learner.py`, `_tuning_e2e.py`, Kaggle enhancement spec Phase 3  
> **Impact**: Optimal combination of diverse model families; typically 1-3% accuracy improvement

---

## Problem Statement

Our current ensemble approach is simple averaging or voting across XGBoost, Random Forest, and Logistic Regression. This treats all models equally, but some models are better in certain regimes (e.g., LR is better calibrated, trees find interactions better). A **stacking meta-learner** learns optimal weights for combining model predictions.

Sports-quant implements a full stacking pipeline with OOF prediction collection, diverse base learners, and a logistic regression meta-learner. Their backtest shows it consistently outperforms simple averaging.

---

## Architecture

```
Layer 1 (Base Learners — trained on temporal CV folds):
  ├── XGBoost ensemble (10-50 models, averaged)
  ├── Logistic Regression (on scaled difference features)
  └── Random Forest (moderate depth, high n_estimators)

Layer 2 (Meta-Learner — trained on OOF predictions from Layer 1):
  └── Logistic Regression (3 inputs → 1 output probability)
```

### Why Logistic Regression as Meta-Learner?

- Only 3 input features (one per base learner) — complex meta-learners would overfit
- Interpretable coefficients show how much each base learner contributes
- Convex optimization guarantee — no local minima
- Sports-quant found this outperforms more complex meta-learners on their data size

---

## Implementation Plan

### Step 1: Collect OOF Predictions

For each temporal CV fold, train all base learners and predict on the held-out year:

```python
BASE_LEARNERS = ("xgb_ensemble", "logistic_regression", "random_forest")

def collect_stacked_oof(matchups_df, prior_years, hyperparams):
    oof_by_learner = {name: [] for name in BASE_LEARNERS}
    all_labels = []
    
    for val_year in prior_years:
        train_years = [y for y in prior_years if y < val_year]
        if len(train_years) < 3:
            continue
        
        X_train, y_train = prepare_fold(matchups_df, train_years)
        X_val, y_val = prepare_fold(matchups_df, [val_year])
        
        # XGBoost ensemble (multiple seeds, averaged)
        xgb_preds = train_xgb_ensemble(X_train, y_train, X_val, n_models=10)
        oof_by_learner["xgb_ensemble"].append(xgb_preds)
        
        # Logistic Regression (with imputation + scaling)
        lr_preds = train_lr(X_train, y_train, X_val)
        oof_by_learner["logistic_regression"].append(lr_preds)
        
        # Random Forest
        rf_preds = train_rf(X_train, y_train, X_val)
        oof_by_learner["random_forest"].append(rf_preds)
        
        all_labels.append(y_val)
    
    matrix = np.column_stack([
        np.concatenate(oof_by_learner[name]) for name in BASE_LEARNERS
    ])
    labels = np.concatenate(all_labels)
    return matrix, labels
```

### Step 2: Train Meta-Learner

```python
def train_meta_learner(oof_matrix, oof_labels, C=1.0):
    meta = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    meta.fit(oof_matrix, oof_labels)
    
    # Log coefficients for interpretability
    for name, coef in zip(BASE_LEARNERS, meta.coef_[0]):
        print(f"  {name}: {coef:.4f}")
    
    return meta
```

### Step 3: Predict on Backtest

```python
def predict_with_stack(meta_model, xgb_preds, lr_preds, rf_preds):
    base_preds = np.column_stack([xgb_preds, lr_preds, rf_preds])
    return meta_model.predict_proba(base_preds)[:, 1]
```

### Step 4: End-to-End Integration

For each backtest year:
1. Train all base learners on prior years
2. Collect OOF predictions for meta-learner training
3. Train meta-learner on OOF
4. Get base learner predictions on backtest year
5. Run meta-learner to get final predictions

### Step 5: Tune Meta-Learner Hyperparameters

Sports-quant's `_tuning_e2e.py` tunes both the base models AND the meta-learner in two phases:
- **Phase 1**: Tune XGBoost params while holding meta-learner fixed
- **Phase 2**: Freeze XGBoost params, tune meta-learner params (C, RF depth, etc.)

Tunable meta-learner params:
```yaml
meta_learner:
  enabled: true
  lr_C: 1.0
  rf_n_estimators: 200
  rf_max_depth: 6
  rf_min_samples_leaf: 20
  meta_C: 1.0
  lgbm_oof_ensemble_size: 10
```

---

## Base Learner Specifications

### XGBoost Ensemble
- Train N models (10-50) with different random seeds
- Average predictions across all models
- Uses tuned hyperparameters from Optuna

### Logistic Regression Pipeline
- `SimpleImputer(strategy="median")` → `StandardScaler()` → `LogisticRegression(C=...)`
- Captures linear trends that trees might miss
- Naturally well-calibrated

### Random Forest Pipeline
- `SimpleImputer(strategy="median")` → `RandomForestClassifier(...)`
- `n_estimators=200`, `max_depth=6`, `min_samples_leaf=20`
- Provides feature bagging diversity

---

## Optional: Additional Base Learners

If we want to go further (sports-quant's Kaggle spec mentions these):

| Learner | What It Adds | Effort |
|---------|-------------|--------|
| Small MLP (2-layer, 32-16) | Non-linear feature interactions | Medium |
| Bradley-Terry model | Principled pairwise comparison | High |
| Elo-based predictor | Trajectory/momentum signal | Medium (needs game-by-game data) |

---

## Acceptance Criteria

- [ ] OOF predictions collected via temporal CV for all base learners
- [ ] Meta-learner trained and coefficients logged
- [ ] Stacked ensemble outperforms simple averaging on backtest
- [ ] End-to-end tuning pipeline with Phase 1 + Phase 2
- [ ] Meta-learner config stored in project config file

---

## Dependencies

- Temporal CV (`roadmap-temporal-cv-and-tuning.md`) — needed for OOF collection
- Difference features (recommended for best base learner performance)
- `scikit-learn` Pipeline and Imputer (already in requirements)
