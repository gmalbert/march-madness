# Roadmap: Automated Testing & CI/CD Pipeline

> **Priority**: P2 — Risk reduction for rapid iteration  
> **Estimated Effort**: Medium  
> **Not in sports-quant analysis** — identified from reviewing repo structure  
> **Impact**: Catch regressions before they corrupt results, enable confident refactoring

---

## Problem Statement

The current repo has 20+ test files scattered in `scripts/` (e.g., `test_barttorvik.py`, `test_model_training.py`, etc.) but no formal test runner, no CI pipeline, and no way to know if changes break existing functionality. Sports-quant has pytest + GitHub Actions CI running on every push, which caught regressions early.

For a project where model integrity is paramount (bad data = meaningless predictions), automated testing is not optional — it's insurance against silent failures.

---

## Part 1: Test Framework

### Structure

```
tests/
├── conftest.py                    # Shared fixtures (sample data, mock models)
├── test_features/
│   ├── test_difference_features.py
│   ├── test_matchup_features.py
│   └── test_symmetrization.py
├── test_data/
│   ├── test_kenpom_loader.py
│   ├── test_barttorvik_loader.py
│   └── test_team_canonicalization.py
├── test_modeling/
│   ├── test_training.py
│   ├── test_calibration.py
│   ├── test_debiasing.py
│   └── test_ensemble.py
├── test_evaluation/
│   ├── test_temporal_cv.py
│   ├── test_metrics.py
│   └── test_backtest.py
└── test_simulation/
    ├── test_bracket_simulation.py
    └── test_upset_detection.py
```

### Critical Tests

**Feature integrity tests:**
```python
def test_difference_features_are_antisymmetric():
    """Swapping teams must negate all odd-symmetry features."""
    features_ab = compute_difference_features(team_a, team_b)
    features_ba = compute_difference_features(team_b, team_a)
    
    for col in ODD_SYMMETRY_FEATURES:
        assert features_ab[col] == pytest.approx(-features_ba[col])

def test_even_features_invariant_under_swap():
    """Even-symmetry features must be identical regardless of team order."""
    features_ab = compute_matchup_features(team_a, team_b)
    features_ba = compute_matchup_features(team_b, team_a)
    
    for col in EVEN_SYMMETRY_FEATURES:
        assert features_ab[col] == pytest.approx(features_ba[col])
```

**Data leakage tests:**
```python
def test_no_temporal_leakage_in_cv():
    """Training data must never include games from validation year or later."""
    for fold in temporal_cv_folds(df):
        train_years = df.loc[fold["train_idx"], "year"].unique()
        val_year = fold["val_year"]
        assert all(y < val_year for y in train_years)

def test_symmetrization_only_in_training():
    """Symmetrization must not be applied to validation or test data."""
    # Verify that val/test set sizes are unchanged
```

**Calibration tests:**
```python
def test_probabilities_clipped():
    """No probability should ever be 0.0 or 1.0."""
    calibrated = calibrate(model_probs)
    assert calibrated.min() >= 0.025
    assert calibrated.max() <= 0.975
```

---

## Part 2: GitHub Actions CI

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'
          cache: pip
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ -v --tb=short
```

### What CI Should Catch

| Check | What It Prevents |
|-------|-----------------|
| Unit tests pass | Logic regressions |
| Feature symmetry tests | Debiasing bugs |
| No temporal leakage | Inflated backtest scores |
| Import checks | Broken module references |
| Linting (optional) | Code style drift |

---

## Part 3: Smoke Tests for Data Pipeline

Quick tests that verify the data pipeline works end-to-end:

```python
def test_kenpom_canonical_has_expected_teams():
    """KenPom canonical file should have 350+ teams."""
    df = pd.read_csv("data_files/kenpom_canonical.csv")
    assert len(df) >= 350

def test_training_data_has_no_nulls_in_target():
    """Training target column must have no missing values."""
    df = pd.read_csv("data_files/training_data_enriched.csv")
    assert df["team1_win"].notna().all()

def test_barttorvik_no_post_tournament_data():
    """BartTorvik data should not reflect tournament game outcomes."""
    # Compare pre-tournament snapshot vs post-season — should be identical
```

---

## Part 4: Regression Test for Model Accuracy

After any code change, verify that backtest accuracy doesn't silently degrade:

```python
# tests/test_regression.py
MINIMUM_ACCURACY = {
    2021: 0.60,  # Set conservative floors
    2022: 0.60,
    2023: 0.60,
    2024: 0.65,
    2025: 0.65,
}

def test_backtest_accuracy_above_floor():
    """Model accuracy should not drop below known floors."""
    results = run_quick_backtest()  # Uses cached models
    for year, accuracy in results.items():
        assert accuracy >= MINIMUM_ACCURACY[year], \
            f"Year {year}: {accuracy:.4f} < floor {MINIMUM_ACCURACY[year]}"
```

---

## Acceptance Criteria

- [ ] `tests/` directory with organized test modules
- [ ] Feature symmetry/antisymmetry verified
- [ ] Temporal leakage prevention verified
- [ ] Probability clipping verified
- [ ] CI pipeline runs on push/PR to main
- [ ] Minimum accuracy regression test in place
- [ ] `pytest` added to `requirements.txt` (dev dependencies)

---

## Dependencies

- Difference features (for symmetry tests)
- Temporal CV (for leakage tests)
- Or: can start with simpler tests today and add more as features are implemented
