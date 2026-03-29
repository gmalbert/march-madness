# Roadmap: Model Configuration & Versioning

> **Priority**: P1 — Needed for reproducibility and systematic improvement  
> **Estimated Effort**: Low  
> **Source**: sports-quant `model_config.yaml`, `_versioning.py`, `_config.py`  
> **Impact**: Reproducible experiments, A/B comparison, clean upgrade path

---

## Problem Statement

Our current repo has hyperparameters scattered across multiple Python files (`model_training.py`, `advanced_model_training.py`, `train_tournament_models.py`, etc.). This makes it impossible to:

1. Compare model versions systematically
2. Reproduce a specific run
3. Know what parameters produced a given result
4. Tune parameters without editing source code

Sports-quant centralizes all configuration in a `model_config.yaml` file with version tracking.

---

## Part 1: Centralized Configuration File

Create `model_config.yaml` at project root:

```yaml
# Model configuration — all hyperparameters and settings in one place
# Version this file in git to track parameter changes over time

model_version: v1

# Data sources
data:
  kenpom: data_files/kenpom_canonical.csv
  barttorvik: data_files/barttorvik_canonical.csv
  training_data: data_files/training_data_enriched.csv

# Feature engineering
features:
  mode: combined                    # "raw", "difference", or "combined"
  symmetrize: true                  # Double training data via row mirroring
  kenpom_features: 10               # Number of KenPom difference features
  barttorvik_features: 12           # Number of BartTorvik difference features
  matchup_features: 11              # Number of matchup interaction features

# XGBoost hyperparameters (tuned via Optuna)
hyperparameters:
  objective: binary:logistic
  eval_metric: logloss
  max_depth: 5
  learning_rate: 0.05
  n_estimators: 500
  min_child_weight: 30
  reg_alpha: 1.0
  reg_lambda: 5.0
  subsample: 0.8
  colsample_bytree: 0.7
  gamma: 0.1

# Training settings
train:
  models_to_train: 50               # Number of models with different seeds
  top_models: 3                     # For detailed analysis (all used in ensemble)
  early_stopping_rounds: 50
  min_boosting_rounds: 50
  tournament_weight: 5              # Weight multiplier for tournament games

# Backtest configuration
backtest:
  years: [2019, 2021, 2022, 2023, 2024, 2025]
  val_years: 2                      # Number of years for validation in early stopping

# Optuna tuning
tuning:
  n_trials: 150
  cv_folds:
    - { train_end: 2020, val_year: 2021 }
    - { train_end: 2021, val_year: 2022 }
    - { train_end: 2022, val_year: 2023 }
    - { train_end: 2023, val_year: 2024 }
    - { train_end: 2024, val_year: 2025 }

# Meta-learner (stacking ensemble)
meta_learner:
  enabled: false
  lr_C: 1.0
  rf_n_estimators: 200
  rf_max_depth: 6
  rf_min_samples_leaf: 20
  meta_C: 1.0

# Calibration
calibration:
  method: isotonic                   # "isotonic" or "platt"
  clip_min: 0.025
  clip_max: 0.975

# Bracket simulation
simulation:
  n_simulations: 10000
  rng_seed: 42
```

---

## Part 2: Config Loading

```python
# config.py — Centralized configuration loader

import yaml
from pathlib import Path

CONFIG_FILE = Path(__file__).parent / "model_config.yaml"

def load_config() -> dict:
    """Load model configuration from YAML file."""
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)

def load_hyperparams() -> dict:
    """Load just the hyperparameters section."""
    return load_config()["hyperparameters"]
```

---

## Part 3: Model Versioning

Sports-quant tracks model versions (v1, v2, ..., v6b) and stores backtest results per version:

```
data_files/backtest/
  v1/
    2021/ (results, models, plots)
    2022/
    ...
    multi_year_summary.txt
  v2/
    ...
```

### Version Tracking

Each time you change features, hyperparameters, or model architecture:
1. Bump `model_version` in config
2. Run backtest — results go to new version directory
3. Compare with previous versions

### Version Comparison Report

```python
def compare_versions(version_a: str, version_b: str):
    """Compare backtest results across two model versions."""
    results_a = load_summary(f"data_files/backtest/{version_a}/multi_year_summary.txt")
    results_b = load_summary(f"data_files/backtest/{version_b}/multi_year_summary.txt")
    
    print(f"{'Year':<8} {version_a:>12} {version_b:>12} {'Delta':>10}")
    for year in results_a["years"]:
        a = results_a["accuracy"][year]
        b = results_b["accuracy"][year]
        print(f"{year:<8} {a:>12.4f} {b:>12.4f} {b-a:>+10.4f}")
```

---

## Part 4: Experiment Tracking

Beyond config versioning, consider lightweight experiment tracking:

| Approach | Complexity | When to Use |
|----------|-----------|-------------|
| Config file + git commits | Low | Current stage |
| CSV log of run results | Low | Once doing many experiments |
| MLflow / Weights & Biases | Medium | If project grows significantly |

For now, config + git + version directories is sufficient.

---

## Acceptance Criteria

- [ ] Single `model_config.yaml` with all tunable parameters
- [ ] `load_config()` function used by all training scripts
- [ ] No hyperparameters hardcoded in training files
- [ ] Model version tracked in config and used for output directory naming
- [ ] Multi-year summary saved for each version
- [ ] Easy comparison between versions

---

## Dependencies

- None — this is infrastructure that enables everything else
