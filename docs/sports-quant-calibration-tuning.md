# Sports-Quant Calibration, Tuning & Meta-Learner

> Source: [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) — March Madness module  
> Purpose: Document probability calibration, Optuna hyperparameter tuning, debiasing, and stacking meta-learner techniques.

---

## Executive Summary

| Technique | Our Current State | sports-quant Approach | Expected Impact |
|-----------|-------------------|----------------------|-----------------|
| Probability calibration | None | Isotonic regression on OOF predictions | High — fixes overconfident probs |
| Hyperparameter tuning | Default XGBoost params | Optuna Bayesian optimization (200 trials) | High — significant log loss improvement |
| Validation strategy | Random train/test split | Temporal CV (train on earlier years, validate on later) | High — prevents data leakage |
| Debiasing | None | Column-swap averaging / difference negation | Medium — eliminates positional bias |
| Stacking ensemble | Simple averaging of XGB+RF+LR | Meta-learner (LR) on OOF predictions of LightGBM+LR+RF | Medium — optimal model combination |
| Early stopping | None | 50-round patience with minimum 50 rounds | Medium — prevents overfitting |

---

## 1. Probability Calibration

Raw model probabilities are often poorly calibrated — a prediction of "70% win" may actually correspond to 60% or 80% real-world win rate. Calibration fixes this.

### Why This Matters
- **Bracket scoring** uses probabilities directly (expected value of advancing teams)
- **Log loss** severely penalizes overconfident wrong predictions
- **Survivor pools** need accurate probabilities for optimal pick selection
- Without calibration, our model's 70% predictions may have very different actual win rates

### Isotonic Regression Calibration

```python
# calibration.py — Probability calibration for tournament predictions

import logging

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


def fit_calibrator(
    oof_probabilities: np.ndarray,
    oof_labels: np.ndarray,
    method: str = "isotonic",
) -> IsotonicRegression | LogisticRegression:
    """Fit a calibration model on out-of-fold predictions.
    
    Uses OOF (out-of-fold) predictions from temporal CV to avoid
    calibrating on the same data the model was trained on.
    
    Args:
        oof_probabilities: Raw model probabilities from temporal CV folds.
        oof_labels: Actual binary outcomes.
        method: "isotonic" for IsotonicRegression, "platt" for Platt scaling.
        
    Returns:
        Fitted calibrator.
    """
    if len(oof_probabilities) != len(oof_labels):
        raise ValueError(
            f"Shape mismatch: probs ({len(oof_probabilities)}) "
            f"vs labels ({len(oof_labels)})"
        )
    
    if len(oof_probabilities) < 10:
        raise ValueError(f"Too few samples for calibration: {len(oof_probabilities)}")
    
    if method == "isotonic":
        calibrator = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip",
        )
        calibrator.fit(oof_probabilities, oof_labels)
    elif method == "platt":
        calibrator = LogisticRegression(C=1.0, solver="lbfgs")
        calibrator.fit(oof_probabilities.reshape(-1, 1), oof_labels)
    else:
        raise ValueError(f"Unknown method: {method!r}")
    
    logger.info("Fitted %s calibrator on %d samples", method, len(oof_probabilities))
    return calibrator


def calibrate_probabilities(
    calibrator: IsotonicRegression | LogisticRegression,
    raw_probabilities: np.ndarray,
    clip_min: float = 0.025,
    clip_max: float = 0.975,
) -> np.ndarray:
    """Apply calibration and clip to safe probability range.
    
    Clipping prevents infinite log loss from predictions of exactly 0 or 1.
    
    Args:
        calibrator: Fitted calibration model.
        raw_probabilities: Uncalibrated model probabilities.
        clip_min: Minimum probability (prevents infinite log loss).
        clip_max: Maximum probability.
        
    Returns:
        Calibrated and clipped probabilities.
    """
    if isinstance(calibrator, IsotonicRegression):
        calibrated = calibrator.predict(raw_probabilities)
    else:
        calibrated = calibrator.predict_proba(
            raw_probabilities.reshape(-1, 1)
        )[:, 1]
    
    return np.clip(calibrated, clip_min, clip_max)


def collect_oof_predictions(
    matchups_df: pd.DataFrame,
    cv_folds: list[dict],
    build_model_fn,
    compute_features_fn,
    target_column: str = "Team1_Win",
    year_column: str = "YEAR",
    do_symmetrize: bool = True,
    early_stop_rounds: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect out-of-fold predictions from temporal CV for calibration.
    
    Trains a model on each fold's training data and predicts on the
    validation fold. Concatenates all fold predictions.
    
    Args:
        matchups_df: Full matchup DataFrame with raw columns.
        cv_folds: List of {"train_end": year, "val_year": year} dicts.
        build_model_fn: Function() -> model (e.g. XGBClassifier).
        compute_features_fn: Function(df) -> feature DataFrame.
        target_column: Name of the target column.
        year_column: Name of the year column.
        do_symmetrize: Whether to symmetrize training data.
        early_stop_rounds: Early stopping patience.
        
    Returns:
        Tuple of (oof_probabilities, oof_labels) as numpy arrays.
    """
    available_years = set(matchups_df[year_column].unique())
    all_probs = []
    all_labels = []
    
    for fold in cv_folds:
        val_year = fold["val_year"]
        train_end = fold["train_end"]
        
        if train_end >= val_year:
            raise ValueError(
                f"Data leakage: train_end ({train_end}) >= val_year ({val_year})"
            )
        
        if val_year not in available_years:
            logger.warning("Skipping fold val_year=%d (no data)", val_year)
            continue
        
        train_raw = matchups_df[matchups_df[year_column] <= train_end]
        val_raw = matchups_df[matchups_df[year_column] == val_year]
        
        if len(val_raw) == 0:
            continue
        
        X_train = compute_features_fn(train_raw)
        y_train = train_raw[target_column].reset_index(drop=True)
        X_val = compute_features_fn(val_raw)
        y_val = val_raw[target_column].reset_index(drop=True)
        
        if do_symmetrize:
            X_train_sym = pd.concat([X_train, X_train * -1], ignore_index=True)
            y_train_sym = pd.concat([y_train, 1 - y_train], ignore_index=True)
            X_train, y_train = X_train_sym, y_train_sym
        
        model = build_model_fn()
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        
        y_val_proba = model.predict_proba(X_val)[:, 1]
        all_probs.append(y_val_proba)
        all_labels.append(y_val.to_numpy())
        
        logger.info(
            "OOF fold val_year=%d: %d samples, mean prob=%.3f",
            val_year, len(y_val), y_val_proba.mean(),
        )
    
    if not all_probs:
        raise ValueError("No OOF predictions collected")
    
    return np.concatenate(all_probs), np.concatenate(all_labels)
```

### Configuration

```yaml
# Add to model config
calibration:
  enabled: true
  method: isotonic       # "isotonic" or "platt"
  clip_min: 0.025        # Prevent infinite log loss
  clip_max: 0.975
```

---

## 2. Optuna Hyperparameter Tuning

Our models use default XGBoost parameters. Optuna can find much better parameters via Bayesian optimization.

### Why This Matters
- Default parameters are generic — not optimized for our specific data
- Tournament prediction data is small (~1000 games) — regularization matters enormously
- Optuna's Bayesian search is far more efficient than grid search

### Implementation

```python
# hyperparameter_tuning.py — Optuna Bayesian optimization

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


def run_optuna_study(
    matchups_df: pd.DataFrame,
    compute_features_fn,
    cv_folds: list[dict],
    n_trials: int = 200,
    do_symmetrize: bool = True,
    early_stop_rounds: int = 50,
    target_column: str = "Team1_Win",
    year_column: str = "YEAR",
) -> dict:
    """Run Bayesian hyperparameter optimization with temporal CV.
    
    Uses Optuna to find optimal XGBoost parameters by minimizing
    average log loss across temporal CV folds.
    
    Args:
        matchups_df: Full matchup DataFrame.
        compute_features_fn: Function(df) -> feature DataFrame.
        cv_folds: List of {"train_end": year, "val_year": year} dicts.
        n_trials: Number of Optuna trials.
        do_symmetrize: Whether to symmetrize training data.
        early_stop_rounds: Early stopping patience.
        
    Returns:
        Dict of best hyperparameters.
    """
    available_years = sorted(matchups_df[year_column].unique().tolist())
    valid_folds = [f for f in cv_folds if f["val_year"] in available_years]
    
    if not valid_folds:
        raise ValueError(f"No valid CV folds. Available years: {available_years}")
    
    logger.info("Starting Optuna: %d trials, %d CV folds", n_trials, len(valid_folds))
    
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 1500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        }
        
        fold_losses = []
        
        for fold in valid_folds:
            train_end = fold["train_end"]
            val_year = fold["val_year"]
            
            if train_end >= val_year:
                raise ValueError(f"Data leakage: train_end ({train_end}) >= val_year ({val_year})")
            
            train_raw = matchups_df[matchups_df[year_column] <= train_end]
            val_raw = matchups_df[matchups_df[year_column] == val_year]
            
            if len(val_raw) == 0:
                continue
            
            X_train = compute_features_fn(train_raw)
            y_train = train_raw[target_column].reset_index(drop=True)
            X_val = compute_features_fn(val_raw)
            y_val = val_raw[target_column].reset_index(drop=True)
            
            if do_symmetrize:
                X_train = pd.concat([X_train, X_train * -1], ignore_index=True)
                y_train = pd.concat([y_train, 1 - y_train], ignore_index=True)
            
            model = XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                early_stopping_rounds=early_stop_rounds,
                **params,
            )
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            
            y_val_proba = model.predict_proba(X_val)[:, 1]
            fold_losses.append(log_loss(y_val, y_val_proba))
        
        if not fold_losses:
            return float("inf")
        
        return float(np.mean(fold_losses))
    
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=n_trials)
    
    logger.info("Best log loss: %.4f", study.best_value)
    logger.info("Best params: %s", study.best_params)
    
    return study.best_params


# Temporal CV fold configuration
# Each fold: train on years <= train_end, validate on val_year
# This prevents temporal data leakage
TEMPORAL_CV_FOLDS = [
    {"train_end": 2013, "val_year": 2014},
    {"train_end": 2015, "val_year": 2016},
    {"train_end": 2017, "val_year": 2018},
    {"train_end": 2018, "val_year": 2019},
    {"train_end": 2021, "val_year": 2022},
    {"train_end": 2022, "val_year": 2023},
    {"train_end": 2023, "val_year": 2024},
    {"train_end": 2024, "val_year": 2025},
]
```

### sports-quant Optimized Parameters (for reference)
These are the Optuna-tuned LightGBM parameters from their `model_config.yaml`:

```yaml
hyperparameters:
  objective: binary
  metric: binary_logloss
  num_leaves: 55
  max_depth: 3
  learning_rate: 0.257
  n_estimators: 177
  min_child_samples: 43
  reg_alpha: 0.303
  reg_lambda: 3.832
  subsample: 0.562
  colsample_bytree: 0.686
  min_split_gain: 0.218
```

Key observations:
- **Very shallow trees** (`max_depth: 3`) — strong regularization for small data
- **Moderate learning rate** (0.257) — not too aggressive
- **Few estimators** (177) — overfitting prevention with early stopping
- **Strong L2 regularization** (`reg_lambda: 3.8`)
- **Heavy feature subsampling** (`colsample_bytree: 0.686`)

---

## 3. Debiasing (Column-Swap Averaging)

Eliminates bias from Team1/Team2 position assignment.

### The Problem
Models can learn a spurious correlation: "Team1 tends to win" (because matchup data may have a pattern in which team is assigned to position 1). This biases predictions.

### With Difference Features (Simple)
For difference features, debiasing is trivial — just negate:

```python
def debias_difference_predictions(
    models: list,
    X_original: pd.DataFrame,
) -> np.ndarray:
    """Debias predictions by averaging original and negated feature predictions.
    
    For difference features: swapping teams = negating all features.
    Average the original prediction with 1 - swapped prediction.
    
    Args:
        models: List of trained models.
        X_original: Original difference features.
        
    Returns:
        Debiased probability predictions.
    """
    # Predictions on original data
    original_probs = [m.predict_proba(X_original)[:, 1] for m in models]
    avg_original = np.mean(original_probs, axis=0)
    
    # Predictions on negated (swapped) data
    X_swapped = X_original * -1
    swapped_probs = [1 - m.predict_proba(X_swapped)[:, 1] for m in models]
    avg_swapped = np.mean(swapped_probs, axis=0)
    
    # Average both
    return (avg_original + avg_swapped) / 2
```

### With Raw Features (Column Swap)
For raw features (Team1 columns + Team2 columns), swap the column groups:

```python
def swap_team_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Swap Team1 and Team2 feature columns for debiasing.
    
    For raw features only. Identifies _Team2 suffixed columns
    and swaps them with their base counterparts.
    """
    pairs = []
    for column in X.columns:
        if column.endswith("_Team2"):
            base = column[:-len("_Team2")]
            if base in X.columns:
                pairs.append((base, column))
    
    swapped_data = {}
    for col1, col2 in pairs:
        swapped_data[col1] = X[col2]
        swapped_data[col2] = X[col1]
    
    return X.assign(**swapped_data)


def debias_raw_predictions(
    models: list,
    X_original: pd.DataFrame,
) -> np.ndarray:
    """Debias using column swap for raw features."""
    original_probs = [m.predict_proba(X_original)[:, 1] for m in models]
    avg_original = np.mean(original_probs, axis=0)
    
    X_swapped = swap_team_columns(X_original)
    swapped_probs = [1 - m.predict_proba(X_swapped)[:, 1] for m in models]
    avg_swapped = np.mean(swapped_probs, axis=0)
    
    return (avg_original + avg_swapped) / 2
```

---

## 4. Stacking Meta-Learner

Trains diverse base models (LightGBM/XGBoost ensemble, Logistic Regression, Random Forest), collects their OOF predictions, and trains a meta-learner to optimally combine them.

### Architecture

```
Base Learners (trained on features):
  ├── XGBoost Ensemble (N models, averaged)
  ├── Logistic Regression (with StandardScaler)
  └── Random Forest

         ↓ OOF predictions (temporal CV)

Meta-Learner (Logistic Regression):
  Input: 3 columns (one per base learner's probability)
  Output: Final calibrated probability
```

### Implementation

```python
# meta_learner.py — Stacking ensemble for tournament predictions

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

BASE_LEARNER_NAMES = ("xgb_ensemble", "logistic_regression", "random_forest")


@dataclass(frozen=True)
class StackedOOF:
    """Out-of-fold predictions from all base learners."""
    matrix: np.ndarray   # (n_samples, n_base_learners)
    labels: np.ndarray   # (n_samples,)
    names: tuple[str, ...]


@dataclass(frozen=True)
class TrainedStack:
    """A trained meta-learner with its base learner predictions."""
    meta_model: LogisticRegression
    base_predictions: np.ndarray  # (n_test, n_base_learners)
    meta_predictions: np.ndarray  # (n_test,)
    names: tuple[str, ...]


def _build_lr(meta_cfg: dict) -> Pipeline:
    """Build imputed + scaled logistic regression pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=meta_cfg.get("lr_C", 1.0),
            solver="lbfgs", max_iter=1000,
        )),
    ])


def _build_rf(meta_cfg: dict) -> Pipeline:
    """Build imputed random forest pipeline."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=meta_cfg.get("rf_n_estimators", 200),
            max_depth=meta_cfg.get("rf_max_depth", 6),
            min_samples_leaf=meta_cfg.get("rf_min_samples_leaf", 20),
            random_state=42,
        )),
    ])


def _train_xgb_ensemble(
    X_train: pd.DataFrame, y_train: pd.Series,
    X_pred: pd.DataFrame,
    hyperparams: dict, n_models: int = 10,
    early_stop_rounds: int = 50,
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
) -> np.ndarray:
    """Train N XGBoost models with different seeds and return averaged predictions."""
    all_probas = []
    rng = np.random.RandomState(42)
    
    for _ in range(n_models):
        seed = int(rng.randint(1, 10000))
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=seed,
            verbosity=0,
            early_stopping_rounds=early_stop_rounds,
            **{k: v for k, v in hyperparams.items() 
               if k not in ("objective", "eval_metric", "metric")},
        )
        
        if X_val is not None and len(X_val) > 0:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            model.fit(X_train, y_train)
        
        all_probas.append(model.predict_proba(X_pred)[:, 1])
    
    return np.mean(all_probas, axis=0)


def collect_stacked_oof(
    matchups_df: pd.DataFrame,
    prior_years: list[int],
    compute_features_fn,
    hyperparams: dict,
    meta_cfg: dict,
    do_symmetrize: bool = True,
    target_column: str = "Team1_Win",
    year_column: str = "YEAR",
    early_stop_rounds: int = 50,
) -> StackedOOF:
    """Collect OOF predictions from all base learners via temporal CV.
    
    For each year with enough training history, trains all base learners
    on earlier years and predicts on that year.
    
    Returns:
        StackedOOF with concatenated predictions from all folds.
    """
    min_train_years = meta_cfg.get("min_oof_years", 3)
    n_xgb = meta_cfg.get("xgb_oof_ensemble_size", 10)
    
    oof_by_learner = {name: [] for name in BASE_LEARNER_NAMES}
    all_labels = []
    
    for val_year in prior_years:
        train_years = [y for y in prior_years if y < val_year]
        if len(train_years) < min_train_years:
            continue
        
        train_df = matchups_df[matchups_df[year_column].isin(train_years)]
        val_df = matchups_df[matchups_df[year_column] == val_year]
        
        if len(val_df) == 0:
            continue
        
        X_train = compute_features_fn(train_df)
        y_train = train_df[target_column].reset_index(drop=True)
        X_val = compute_features_fn(val_df)
        y_val = val_df[target_column].reset_index(drop=True)
        
        if do_symmetrize:
            X_train_s = pd.concat([X_train, X_train * -1], ignore_index=True)
            y_train_s = pd.concat([y_train, 1 - y_train], ignore_index=True)
        else:
            X_train_s, y_train_s = X_train, y_train
        
        # XGBoost ensemble
        xgb_preds = _train_xgb_ensemble(
            X_train_s, y_train_s, X_val, hyperparams,
            n_models=n_xgb, early_stop_rounds=early_stop_rounds,
        )
        oof_by_learner["xgb_ensemble"].append(xgb_preds)
        
        # Logistic Regression
        lr_model = _build_lr(meta_cfg)
        lr_model.fit(X_train_s, y_train_s)
        oof_by_learner["logistic_regression"].append(
            lr_model.predict_proba(X_val)[:, 1]
        )
        
        # Random Forest
        rf_model = _build_rf(meta_cfg)
        rf_model.fit(X_train_s, y_train_s)
        oof_by_learner["random_forest"].append(
            rf_model.predict_proba(X_val)[:, 1]
        )
        
        all_labels.append(y_val.to_numpy())
        logger.info("OOF fold val_year=%d: %d samples", val_year, len(y_val))
    
    if not all_labels:
        raise ValueError("No OOF predictions — not enough years")
    
    matrix = np.column_stack([
        np.concatenate(oof_by_learner[name]) for name in BASE_LEARNER_NAMES
    ])
    
    return StackedOOF(
        matrix=matrix,
        labels=np.concatenate(all_labels),
        names=BASE_LEARNER_NAMES,
    )


def train_meta_learner(oof: StackedOOF, meta_cfg: dict) -> LogisticRegression:
    """Train the meta-learner on stacked OOF predictions.
    
    Uses logistic regression — simple enough to avoid overfitting
    on the small stacked feature space (3 features).
    """
    meta_C = meta_cfg.get("meta_C", 1.0)
    meta = LogisticRegression(C=meta_C, solver="lbfgs", max_iter=1000)
    meta.fit(oof.matrix, oof.labels)
    
    coef_dict = {
        name: round(float(c), 4)
        for name, c in zip(oof.names, meta.coef_[0])
    }
    logger.info(
        "Meta-learner: coefficients=%s, intercept=%.4f",
        coef_dict, float(meta.intercept_[0]),
    )
    return meta


def train_and_predict_stack(
    matchups_df: pd.DataFrame,
    backtest_year: int,
    compute_features_fn,
    hyperparams: dict,
    meta_cfg: dict,
    do_symmetrize: bool = True,
    target_column: str = "Team1_Win",
    year_column: str = "YEAR",
    early_stop_rounds: int = 50,
    xgb_backtest_probas: np.ndarray | None = None,
) -> TrainedStack:
    """End-to-end: collect OOF, train meta-learner, predict on backtest year.
    
    Reuses pre-computed XGBoost predictions if provided, trains LR and RF
    on all prior years for the other base inputs.
    
    Returns:
        TrainedStack with meta-learner predictions.
    """
    available_years = sorted(matchups_df[year_column].unique().tolist())
    prior_years = [y for y in available_years if y < backtest_year]
    
    # Step 1: Collect OOF predictions
    oof = collect_stacked_oof(
        matchups_df=matchups_df,
        prior_years=prior_years,
        compute_features_fn=compute_features_fn,
        hyperparams=hyperparams,
        meta_cfg=meta_cfg,
        do_symmetrize=do_symmetrize,
        target_column=target_column,
        year_column=year_column,
        early_stop_rounds=early_stop_rounds,
    )
    
    # Step 2: Train meta-learner
    meta_model = train_meta_learner(oof, meta_cfg)
    
    # Step 3: Get base learner predictions on backtest year
    all_prior_df = matchups_df[matchups_df[year_column].isin(prior_years)]
    backtest_df = matchups_df[matchups_df[year_column] == backtest_year]
    
    X_all_prior = compute_features_fn(all_prior_df)
    y_all_prior = all_prior_df[target_column].reset_index(drop=True)
    X_backtest = compute_features_fn(backtest_df)
    
    if do_symmetrize:
        X_all_prior_s = pd.concat([X_all_prior, X_all_prior * -1], ignore_index=True)
        y_all_prior_s = pd.concat([y_all_prior, 1 - y_all_prior], ignore_index=True)
    else:
        X_all_prior_s, y_all_prior_s = X_all_prior, y_all_prior
    
    # Train LR and RF on all prior data
    lr_model = _build_lr(meta_cfg)
    lr_model.fit(X_all_prior_s, y_all_prior_s)
    lr_preds = lr_model.predict_proba(X_backtest)[:, 1]
    
    rf_model = _build_rf(meta_cfg)
    rf_model.fit(X_all_prior_s, y_all_prior_s)
    rf_preds = rf_model.predict_proba(X_backtest)[:, 1]
    
    # Use pre-computed XGB predictions or train new
    if xgb_backtest_probas is None:
        xgb_backtest_probas = _train_xgb_ensemble(
            X_all_prior_s, y_all_prior_s, X_backtest, hyperparams,
        )
    
    # Step 4: Stack and predict
    base_preds = np.column_stack([xgb_backtest_probas, lr_preds, rf_preds])
    meta_preds = meta_model.predict_proba(base_preds)[:, 1]
    
    logger.info(
        "Meta-learner for %d: mean=%.3f (xgb=%.3f, lr=%.3f, rf=%.3f)",
        backtest_year, meta_preds.mean(),
        xgb_backtest_probas.mean(), lr_preds.mean(), rf_preds.mean(),
    )
    
    return TrainedStack(
        meta_model=meta_model,
        base_predictions=base_preds,
        meta_predictions=meta_preds,
        names=BASE_LEARNER_NAMES,
    )
```

### Meta-Learner Configuration

```yaml
# Add to model config
meta_learner:
  enabled: true
  xgb_oof_ensemble_size: 10   # N XGBoost models per OOF fold
  lr_C: 1.0                   # Logistic Regression regularization
  rf_n_estimators: 200         # Random Forest trees
  rf_max_depth: 6              # RF max depth
  rf_min_samples_leaf: 20      # RF minimum leaf samples
  meta_C: 1.0                 # Meta-learner LR regularization
  min_oof_years: 3             # Minimum training years for OOF
```

---

## 5. Temporal Cross-Validation

Our current approach uses random train/test splits, which allows future games to leak into training data. Temporal CV fixes this:

```python
# Temporal CV folds: train on years <= train_end, validate on val_year
TEMPORAL_CV_FOLDS = [
    {"train_end": 2013, "val_year": 2014},
    {"train_end": 2015, "val_year": 2016},
    {"train_end": 2017, "val_year": 2018},
    {"train_end": 2018, "val_year": 2019},
    {"train_end": 2021, "val_year": 2022},  # Skip 2020 (no tournament)
    {"train_end": 2022, "val_year": 2023},
    {"train_end": 2023, "val_year": 2024},
    {"train_end": 2024, "val_year": 2025},
]
```

**Key rule:** `train_end` must be strictly less than `val_year`. The model never sees any data from the validation year or later during training.

---

## 6. Upset Analysis

Track which upsets models predict and how accurately:

```python
# upset_analysis.py — Track upset predictions across models

import numpy as np
import pandas as pd


def analyze_upsets(
    actual_results: np.ndarray,
    predicted_results: np.ndarray,
    team_data: pd.DataFrame,
) -> dict:
    """Analyze upset predictions vs actual outcomes.
    
    An upset = lower seed (higher number) beats higher seed (lower number).
    
    Args:
        actual_results: Actual outcomes (1=Team1 wins).
        predicted_results: Predicted outcomes.
        team_data: DataFrame with Seed1, Seed2, Team1, Team2, YEAR.
        
    Returns:
        Dict with upset counts, accuracy, and individual upset details.
    """
    results = {
        "total_games": len(actual_results),
        "total_upsets_actual": 0,
        "total_upsets_predicted": 0,
        "correct_upset_predictions": 0,
        "upsets": [],
    }
    
    for i, (actual, pred) in enumerate(zip(actual_results, predicted_results)):
        row = team_data.iloc[i]
        seed1, seed2 = row["Seed1"], row["Seed2"]
        
        # Actual upset: lower-seeded team (higher seed number) won
        is_actual_upset = (
            (seed1 < seed2 and actual == 0) or
            (seed2 < seed1 and actual == 1)
        )
        if is_actual_upset:
            results["total_upsets_actual"] += 1
        
        # Predicted upset: model picked the lower-seeded team
        is_predicted_upset = (
            (seed1 < seed2 and pred == 0) or
            (seed2 < seed1 and pred == 1)
        )
        if is_predicted_upset:
            results["total_upsets_predicted"] += 1
            seed_diff = abs(seed1 - seed2)
            
            underdog = row["Team2"] if seed1 < seed2 else row["Team1"]
            favorite = row["Team1"] if seed1 < seed2 else row["Team2"]
            
            results["upsets"].append({
                "year": row.get("YEAR", ""),
                "underdog": underdog,
                "underdog_seed": max(seed1, seed2),
                "favorite": favorite,
                "favorite_seed": min(seed1, seed2),
                "correctly_predicted": (pred == actual),
            })
        
        if is_actual_upset and is_predicted_upset and pred == actual:
            results["correct_upset_predictions"] += 1
    
    return results
```

---

## Priority Implementation Order

1. **Temporal CV** — Replace random splits, prevent temporal leakage
2. **Optuna tuning** — Optimize XGBoost parameters for our data
3. **Probability calibration** — Fix overconfident predictions
4. **Debiasing** — Eliminate Team1/Team2 positional bias
5. **Stacking meta-learner** — Optimal model combination
6. **Upset analysis** — Track and improve upset prediction accuracy
