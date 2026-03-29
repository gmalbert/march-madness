# Roadmap: Sample Weighting & Training Data Strategy

> **Priority**: P2 — Meaningful improvement for tournament-specific accuracy  
> **Estimated Effort**: Low  
> **Source**: sports-quant Kaggle spec Phase 3.1, own feature_engineering.py analysis  
> **Impact**: Teaches the model tournament-specific patterns without losing regular season signal

---

## Problem Statement

We already weight tournament games 5x in `feature_engineering.py`, which is smart. But our weighting strategy can be refined based on sports-quant's Kaggle research and the Odds Gods model's approach.

---

## Current State

In `feature_engineering.py`:
```python
def build_weighted_training_dataset(...):
    # Tournament games weighted 5x
```

This is directionally correct but can be improved.

---

## Refined Weighting Strategy

### The Odds Gods Approach (0.54 log loss, 77.6% accuracy)

```
Regular season early (days 1-100):        weight = 1
Regular season late + conf tournament:    weight = 2
NCAA tournament games:                    weight = 6
```

### Rationale

Tournament games ARE the distribution we're predicting on. The model should:
1. Learn general basketball patterns from regular season (weight=1)
2. Give more attention to high-stakes late-season games (weight=2)
3. Heavily prioritize tournament-specific dynamics (weight=6)

### Implementation

```python
def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights based on game type and timing."""
    weights = np.ones(len(df))
    
    for i, row in df.iterrows():
        if row["game_type"] == "tournament":
            weights[i] = 6.0
        elif row["game_type"] == "conference_tournament":
            weights[i] = 2.0
        elif row["day_of_season"] > 100:
            weights[i] = 2.0
        else:
            weights[i] = 1.0
    
    return weights
```

### Passing to Models

```python
# XGBoost
model.fit(X_train, y_train, sample_weight=sample_weights)

# LightGBM
model.fit(X_train, y_train, sample_weight=sample_weights)

# Scikit-learn
model.fit(X_train, y_train, sample_weight=sample_weights)
```

---

## Data Strategy: Tournament-Only vs Full Season

### Option A: Tournament-Only Training
- Pro: Small, focused dataset; every game is relevant
- Con: Only ~63 games/year → very small training set
- When: Use if data goes back 20+ years (2003-present = ~1,200 tournament games)

### Option B: Full Season Training  
- Pro: ~16,000 games → much more data
- Con: Regular season dynamics differ from tournament
- When: Use if data is limited years or if sample weighting is effective

### Option C: Hybrid (Recommended)
- Train on full season with sample weights
- Fine-tune on tournament-only data (warm start)
- This is what our `train_tournament_models.py` does — keep this approach but add proper weighting

---

## Ensemble Model Count: All vs Top-N

### Current Approach (Both Repos)
Train 50 models with different random seeds, pick top 3 by validation F1.

### Problem
Picking top 3 by val F1 is overfitting to the validation set. With small data, val F1 variance across seeds is largely noise.

### sports-quant's Solution
**Average ALL 50 models** instead of cherry-picking 3:

```python
# Instead of: top_3_probs = mean([top_models[i].predict_proba() for i in range(3)])
# Use:        all_probs = mean([all_models[i].predict_proba() for i in range(50)])
```

Sports-quant saves all models and uses the full ensemble for final predictions. The top-3 selection is only used for detailed per-model analysis and plots.

### Alternative: Soft Threshold
If some models are genuinely bad, use a soft threshold:
```python
# Average all models with val F1 > median
median_f1 = np.median([m["val_f1"] for m in all_models])
good_models = [m for m in all_models if m["val_f1"] >= median_f1]
```

---

## Acceptance Criteria

- [ ] Sample weighting by game type (regular season, conference tourney, NCAA tournament)
- [ ] Weights configurable in `model_config.yaml`
- [ ] Ensemble uses all 50 models (or soft-filtered subset)
- [ ] Backtest compares top-3 vs all-50 ensemble
- [ ] Sample weights passed correctly to all model types (XGBoost, RF, LR)

---

## Dependencies

- Game-by-game data (needed for day-of-season and conference tournament identification)
- Or: game type labels in training data (simpler, can be done now)
