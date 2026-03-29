# Roadmap: Probability Calibration & Evaluation Metrics

> **Priority**: P1 — Critical for bracket simulation and betting accuracy  
> **Estimated Effort**: Low-Medium  
> **Source**: sports-quant `_calibration.py`, Kaggle enhancement spec Phase 1.5 + 4.x  
> **Impact**: Fixes overconfident probabilities, improves log loss, makes bracket sims and betting reliable

---

## Problem Statement

Our raw XGBoost/LR/RF probabilities are **not calibrated**. A prediction of "75% win probability" may correspond to only 60% real-world win rate. This has cascading effects:

1. **Bracket simulation** — Monte Carlo simulations are only as good as the input probabilities. If we systematically overvalue favorites, our simulated brackets are biased
2. **Value betting** — `find_value_bets()` compares model probability to implied odds. Bad probabilities = bad bets
3. **Upset detection** — Overconfident models never flag upsets because they assign >90% to favorites in nearly all matchups
4. **Log loss** — Overconfident wrong predictions are severely penalized

Sports-quant uses **isotonic regression** on out-of-fold predictions with probability clipping.

---

## Part 1: Isotonic Regression Calibration

### How It Works

1. During temporal CV, collect OOF (out-of-fold) predictions across all folds
2. Fit an `IsotonicRegression` mapping raw probabilities → actual outcomes
3. Apply as a post-processing step on all future predictions
4. Clip final probabilities to `[0.025, 0.975]` — a 16-seed beat a 1-seed in 2018; predicting P=0 yields infinite log loss

### Implementation

```python
from sklearn.isotonic import IsotonicRegression

def fit_calibrator(oof_probs: np.ndarray, oof_labels: np.ndarray):
    calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    calibrator.fit(oof_probs, oof_labels)
    return calibrator

def calibrate(calibrator, raw_probs, clip_min=0.025, clip_max=0.975):
    calibrated = calibrator.predict(raw_probs)
    return np.clip(calibrated, clip_min, clip_max)
```

### OOF Collection Process

```python
def collect_oof_predictions(matchups_df, cv_folds, hyperparams):
    all_probs, all_labels = [], []
    for fold in cv_folds:
        train = matchups_df[matchups_df["year"] <= fold["train_end"]]
        val = matchups_df[matchups_df["year"] == fold["val_year"]]
        
        model = train_model(train, hyperparams)
        probs = model.predict_proba(val_features)[:, 1]
        
        all_probs.append(probs)
        all_labels.append(val_labels)
    
    return np.concatenate(all_probs), np.concatenate(all_labels)
```

---

## Part 2: Better Evaluation Metrics

### Current State

We track:
- MAE (spread/total regression)
- Accuracy (moneyline classification)
- ROI (betting)

### What to Add

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Log Loss** | Quality of probability estimates | Kaggle primary metric; punishes overconfidence |
| **Brier Score** | Mean squared error of probabilities | Easier to interpret than log loss |
| **ESPN Bracket Score** | Weighted bracket points (10/20/40/80/160/320) | Actual bracket pool scoring |
| **Reliability Diagram** | Predicted vs actual win rate in bins | Visual calibration check |
| **Calibration Error (ECE)** | Expected calibration error | Single-number calibration quality |

### ESPN Bracket Scoring

Accuracy treats all rounds equally, but bracket pools don't. Implement standard scoring:

```python
ROUND_POINTS = {
    "R64": 10, "R32": 20, "S16": 40, "E8": 80, "F4": 160, "NCG": 320
}

def espn_bracket_score(predictions, actuals, rounds):
    total = 0
    for pred, actual, round_name in zip(predictions, actuals, rounds):
        if pred == actual:
            total += ROUND_POINTS[round_name]
    return total
```

### Reliability Diagram

```python
def plot_reliability_diagram(y_true, y_proba, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers, bin_actuals = [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_proba >= lo) & (y_proba < hi)
        if mask.sum() > 0:
            bin_centers.append(y_proba[mask].mean())
            bin_actuals.append(y_true[mask].mean())
    # Plot bin_centers vs bin_actuals; perfect calibration = diagonal
```

---

## Part 3: Favorites Benchmark

Sports-quant computes a "always pick favorites" baseline for every backtest year:

```python
favorites_correct = sum(
    (seed1 <= seed2) & (team1_won == 1) |
    (seed1 > seed2) & (team1_won == 0)
)
favorites_accuracy = favorites_correct / total_games
improvement = model_accuracy - favorites_accuracy
```

This is essential context — a model must beat the naive "pick all favorites" strategy to be worth anything.

---

## Acceptance Criteria

- [ ] Isotonic regression calibrator fitted on OOF predictions
- [ ] Probability clipping to [0.025, 0.975]
- [ ] Log loss and Brier score reported alongside accuracy for all evaluations
- [ ] ESPN-style bracket scoring implemented and tracked
- [ ] Reliability diagram generated for each backtest year
- [ ] Favorites baseline computed for every evaluation
- [ ] No predicted probability is ever 0.0 or 1.0

---

## Dependencies

- Temporal CV (`roadmap-temporal-cv-and-tuning.md`) — needed for OOF predictions
- Bracket simulation updates — needed for ESPN scoring integration
