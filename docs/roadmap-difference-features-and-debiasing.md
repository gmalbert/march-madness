# Roadmap: Difference Features & Positional Debiasing

> **Priority**: P0 — Highest ROI single change  
> **Estimated Effort**: Medium  
> **Source**: sports-quant `_features.py`, `_feature_builder.py`, `_debiasing.py`  
> **Impact**: Eliminates positional bias, halves feature dimensionality, enables data symmetrization

---

## Problem Statement

Our current pipeline feeds raw efficiency stats as separate columns (e.g., `team_net_rtg`, `opponent_net_rtg`) and computes diffs ad-hoc in `feature_engineering.py`. This causes:

1. **Positional bias** — models learn "Team1 usually wins" from column ordering
2. **Redundant features** — 36 raw columns when 18 differences carry the same signal
3. **No symmetrization** — can't trivially double training data
4. **Debiasing is impossible** — no clean way to swap team perspectives

Sports-quant moved to difference features and saw their biggest accuracy jump (68% → 72+%). Their model analysis explicitly calls this "the single highest-ROI change."

---

## Implementation Plan

### Step 1: Define Canonical Difference Feature Set

Create `difference_features.py` with explicit definitions:

```
KenPom diffs (10):
  rank_diff, adjEM_diff, adjO_diff, adjO_rank_diff, adjD_diff,
  adjD_rank_diff, adjT_diff, luck_diff, sos_adjEM_diff, ncsos_adjEM_diff

BartTorvik diffs (12):
  bart_rank_diff, bart_adjOE_diff, bart_adjDE_diff, bart_barthag_diff,
  bart_adjT_diff, bart_sos_diff, bart_ncsos_diff, bart_elite_sos_diff,
  bart_wab_diff, bart_qualO_diff, bart_qualD_diff, bart_qual_barthag_diff

Derived features (3):
  seed_diff, efficiency_ratio_diff, seed_x_adjEM_interaction

Total: 25 difference features (vs current ~11 raw)
```

### Step 2: Build Difference Feature Computation

In `feature_engineering.py` or a new `difference_features.py`:

- `compute_kenpom_diffs(row) -> dict` — 10 KenPom differences
- `compute_barttorvik_diffs(row) -> dict` — 12 BartTorvik differences
- `compute_derived_diffs(row) -> dict` — seed_diff, efficiency_ratio_diff, seed_x_adjEM_interaction
- `compute_all_difference_features(df) -> DataFrame` — full pipeline

### Step 3: Implement Data Symmetrization

For each game (TeamA vs TeamB, label=1), add mirror row (TeamB vs TeamA, label=0):
- With difference features, mirroring = negating all features and flipping label
- Doubles training data from ~1,500 to ~3,000 games
- Applies to training data only (never test/validation)

```python
def symmetrize_training_data(X: pd.DataFrame, y: pd.Series):
    X_mirror = X * -1
    y_mirror = 1 - y
    return pd.concat([X, X_mirror]), pd.concat([y, y_mirror])
```

### Step 4: Add Debiasing Layer

Two-phase debiasing (from sports-quant `_debiasing.py`):

1. **Difference negation** — predict on original and negated features, average probabilities
2. **Column-swap** — for any remaining raw features, swap Team1/Team2 columns

```python
def run_debiased_prediction(models, X_original):
    X_swapped = X_original * -1  # for difference features
    orig_probs = np.mean([m.predict_proba(X_original)[:, 1] for m in models], axis=0)
    swap_probs = np.mean([1 - m.predict_proba(X_swapped)[:, 1] for m in models], axis=0)
    return (orig_probs + swap_probs) / 2
```

### Step 5: Update Training Pipeline

- `model_training.py` — use difference features instead of raw
- `train_tournament_models.py` — same update
- `advanced_model_training.py` — same update
- Retrain all models and compare metrics

---

## Acceptance Criteria

- [ ] Difference feature definitions in a dedicated module
- [ ] `compute_all_difference_features()` produces correct output for test cases
- [ ] Symmetrization doubles training set without data leakage
- [ ] Debiased predictions show no positional bias in backtest
- [ ] Backtest accuracy improves over raw-feature baseline
- [ ] All existing prediction pipelines updated to use new features

---

## Dependencies

- BartTorvik Time Machine API (from `roadmap-data-leakage-prevention.md`) — needed for clean historical data
- Temporal CV (`roadmap-temporal-cv-and-tuning.md`) — needed to validate improvement properly

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking existing model pipeline | Medium | Keep raw features as fallback, A/B test both |
| Symmetrization leaks into validation | High | Symmetrize only within training folds |
| Fewer total features than current approach | Low | Difference features are strictly more informative per feature |
