# Roadmap: LightGBM Migration & Model Exploration

> **Priority**: P2 — Moderate improvement, natural fit with other changes  
> **Estimated Effort**: Low  
> **Source**: sports-quant uses LightGBM exclusively for March Madness  
> **Impact**: Faster training, native NaN handling, native categorical support

---

## Problem Statement

Our current ensemble uses XGBoost + Random Forest + Logistic Regression. Sports-quant uses LightGBM as its primary tree-based model. While both are gradient-boosted trees, LightGBM has several practical advantages for our dataset:

---

## Why Consider LightGBM

| Feature | XGBoost | LightGBM | Advantage |
|---------|---------|----------|-----------|
| Training speed | Moderate | 2-10x faster | LightGBM — matters for 50-model ensembles |
| NaN handling | Learns split direction | Native NaN support | LightGBM — cleaner for missing ratings |
| Categorical features | Must encode | Native support | LightGBM — conference, seed as categorical |
| Leaf-wise growth | Level-wise default | Leaf-wise default | LightGBM — better accuracy/depth tradeoff for small data |
| Optuna integration | Good | Excellent | Tie |

### Key Point
LightGBM's leaf-wise growth is particularly advantageous for our small dataset (~1,500-3,000 games). It can fit the data better with fewer leaves, reducing overfitting risk.

---

## Implementation Plan

### Step 1: Add LightGBM to Dependencies

```
# requirements.txt
lightgbm>=4.0.0
```

### Step 2: LightGBM Wrapper Function

```python
from lightgbm import LGBMClassifier

def build_lgbm(hyperparams: dict, random_seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        objective=hyperparams.get("objective", "binary"),
        metric=hyperparams.get("metric", "binary_logloss"),
        random_state=random_seed,
        verbose=-1,
        **{k: v for k, v in hyperparams.items() if k not in ("objective", "metric")},
    )
```

### Step 3: A/B Test vs XGBoost

Don't replace XGBoost — add LightGBM as an ensemble member:

```python
# In stacking meta-learner:
base_learners = {
    "xgboost_ensemble": train_xgb_ensemble(...),
    "lightgbm_ensemble": train_lgbm_ensemble(...),
    "logistic_regression": train_lr(...),
    "random_forest": train_rf(...),
}
```

Let the meta-learner learn the optimal weight. If LightGBM consistently outperforms XGBoost, consider switching the primary model.

### Step 4: LightGBM-Specific Hyperparameters

```yaml
# model_config.yaml — LightGBM hyperparameters
lightgbm_hyperparameters:
  objective: binary
  metric: binary_logloss
  num_leaves: 20          # Lower than default 31 for small data
  max_depth: 6
  learning_rate: 0.05
  n_estimators: 500
  min_child_samples: 30   # Higher for small data
  reg_alpha: 1.0
  reg_lambda: 5.0
  subsample: 0.8
  colsample_bytree: 0.7
  min_split_gain: 0.1
```

---

## Further Model Exploration

Beyond LightGBM, the stacking ensemble could include:

### Bradley-Terry Model
A principled pairwise comparison model that estimates latent team strengths:
- Naturally models "who beats whom"
- Provides uncertainty quantification
- Available via `scipy.optimize` (ML) or PyMC (Bayesian)
- sports-quant mentions this as a high-value addition

### Small Neural Network (MLP)
A 2-layer MLP (32-16 units) could capture non-linear feature interactions:
- Different inductive bias from trees (smooth decision boundaries)
- Works well with difference features
- Low effort with scikit-learn's `MLPClassifier`

### CatBoost
Another gradient boosting library with native categorical support and built-in regularization. Could replace or complement LightGBM.

---

## Acceptance Criteria

- [ ] LightGBM added to requirements and usable in training pipeline
- [ ] A/B comparison of LightGBM vs XGBoost on identical features and CV setup
- [ ] LightGBM-specific hyperparameters tunable via Optuna
- [ ] LightGBM optionally available as a stacking base learner
- [ ] Decision documented: switch primary model or keep both

---

## Dependencies

- `lightgbm` package installation
- Temporal CV (for fair comparison)
- Stacking ensemble (to use as additional base learner)
