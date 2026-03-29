# Roadmap: Temporal Cross-Validation & Optuna Hyperparameter Tuning

> **Priority**: P0 — Fixes fundamental evaluation and training flaws  
> **Estimated Effort**: Medium  
> **Source**: sports-quant `_tuning.py`, `backtest.py`, `_tuning_e2e.py`  
> **Impact**: Prevents data leakage in validation, significantly improves model parameters

---

## Problem Statement

### Issue 1: Random Train/Test Split

Our current pipeline uses random splits (e.g., `train_test_split(test_size=0.2)`). This means a 2023 game could be in training while a 2018 game is in validation. Since CBB performance varies year-to-year, this leaks temporal information and makes validation scores artificially optimistic.

### Issue 2: Default Hyperparameters

Our XGBoost/RF/LR models use near-default parameters. With ~1,500-3,000 training games (tournament-only data is even smaller), defaults almost certainly overfit. Sports-quant's Kaggle analysis shows that Optuna tuning with temporal CV is the second-highest-ROI change after difference features.

---

## Part 1: Temporal Cross-Validation

### Design

Replace random splits with **leave-year-out** temporal CV:

```
Fold 1: Train 2016-2020, Validate 2021
Fold 2: Train 2016-2021, Validate 2022
Fold 3: Train 2016-2022, Validate 2023
Fold 4: Train 2016-2023, Validate 2024
Fold 5: Train 2016-2024, Validate 2025
```

Key rules:
- **Never train on data from the validation year or later**
- Use previous 1-2 years as held-out validation for early stopping
- Backtest year is completely held out
- No 2020 tournament year (COVID) — skip or use as pure training

### Multi-Year Validation for Early Stopping

Sports-quant uses the most recent 2 years before the backtest year as a validation set for early stopping. This gives a larger, more stable validation set than a single 63-game year:

```python
def get_val_years(backtest_year, available_years, n=2):
    candidates = sorted([y for y in available_years if y < backtest_year], reverse=True)
    return candidates[:n]
```

### Implementation

Modify `model_training.py` and `train_tournament_models.py`:

```python
def temporal_cv_folds(df: pd.DataFrame, min_train_years: int = 3):
    """Generate temporal CV folds."""
    years = sorted(df["year"].unique())
    folds = []
    for i, val_year in enumerate(years):
        train_years = [y for y in years if y < val_year]
        if len(train_years) >= min_train_years:
            folds.append({"train_years": train_years, "val_year": val_year})
    return folds
```

---

## Part 2: Optuna Hyperparameter Tuning

### XGBoost Search Space (matching sports-quant's LightGBM approach)

```python
def xgboost_objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
        "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
    }
    
    fold_losses = []
    for fold in temporal_cv_folds:
        # Train on older years, validate on newer
        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                  early_stopping_rounds=50)
        y_proba = model.predict_proba(X_val)[:, 1]
        fold_losses.append(log_loss(y_val, y_proba))
    
    return np.mean(fold_losses)
```

### CV Folds Configuration

Store in a config file (like sports-quant's `model_config.yaml`):

```yaml
tuning:
  n_trials: 150
  cv_folds:
    - { train_end: 2020, val_year: 2021 }
    - { train_end: 2021, val_year: 2022 }
    - { train_end: 2022, val_year: 2023 }
    - { train_end: 2023, val_year: 2024 }
    - { train_end: 2024, val_year: 2025 }
```

### Optimization Objective

- **Primary**: Minimize mean log loss across CV folds (same as Kaggle)
- Not accuracy — accuracy throws away probability information
- Not F1 — F1 is threshold-dependent

### Early Stopping

Sports-quant adds a guard against under-trained models:

```python
# If model stops too early (< 50 rounds), retrain with floor
MIN_BOOSTING_ROUNDS = 50
if model.best_iteration_ < MIN_BOOSTING_ROUNDS:
    model = retrain_with_floor(params, MIN_BOOSTING_ROUNDS)
```

### Save Best Parameters

After tuning, save to config (not hardcoded):

```python
def save_best_params(params: dict, config_path: str):
    config = yaml.safe_load(open(config_path))
    config["hyperparameters"].update(params)
    yaml.dump(config, open(config_path, "w"))
```

---

## Part 3: End-to-End Retune Pipeline

Sports-quant has a sophisticated `retune_e2e.py` script that:

1. Evaluates **current** params through the full pipeline (including meta-learner)
2. Runs Phase 1: Tunes base model (LightGBM) hyperparameters
3. Runs Phase 2: Tunes meta-learner hyperparameters
4. Evaluates **new** params through the full pipeline
5. Prints per-fold comparison table
6. Only saves if improvement exceeds a threshold (0.005 log loss)

We should build a similar A/B comparison workflow:

```python
# retune.py
IMPROVEMENT_THRESHOLD = 0.005

old_results = evaluate_with_current_params(cv_folds)
new_params = run_optuna_tuning(cv_folds, n_trials=150)
new_results = evaluate_with_new_params(cv_folds, new_params)

delta = old_results["mean_log_loss"] - new_results["mean_log_loss"]
if delta > IMPROVEMENT_THRESHOLD:
    save_best_params(new_params)
    print(f"New params saved (improvement: {delta:+.4f})")
else:
    print(f"No improvement ({delta:+.4f} < threshold)")
```

---

## Acceptance Criteria

- [ ] Temporal CV replaces random splits in all training pipelines
- [ ] No validation data comes from the same year as or later than training data
- [ ] Optuna study runs with 100+ trials using temporal CV
- [ ] Best params saved to a config file (not hardcoded in source)
- [ ] Early stopping with minimum rounds guard implemented
- [ ] Log loss tracked as primary optimization metric
- [ ] Retune script with A/B comparison and improvement threshold

---

## Dependencies

- `optuna` added to `requirements.txt`
- `pyyaml` for config management (or similar)
- Difference features (recommended but not required)

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fewer training samples per fold | Medium | Use symmetrization to double data |
| Optuna overfitting to CV folds | Low | Small search space, low trial count relative to parameters |
| Config file management complexity | Low | Start with simple YAML, can evolve to more structured approach |
