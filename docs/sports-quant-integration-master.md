# Sports-Quant Integration Master Plan

> Source: [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant)  
> Purpose: Prioritized integration plan linking all sports-quant analysis documents. Maps each capability to our codebase and provides a phased implementation roadmap.

---

## Our Current State vs. Sports-Quant

| Capability | Our Repo | Sports-Quant | Gap |
|-----------|----------|-------------|-----|
| **Features** | ~11 raw efficiency diffs | 45 features (21 KenPom + 13 BartTorvik + 11 matchup) | **Critical** |
| **Model** | XGBoost + RF + LR ensemble | LightGBM + stacking meta-learner | Moderate |
| **Calibration** | None | Isotonic regression (OOF) | **Critical** |
| **Tuning** | Manual hyperparams | Optuna (200 trials, Bayesian) | **High** |
| **Debiasing** | None | Column-swap + difference negation | High |
| **Data symmetry** | None | Row duplication with label flip | High |
| **Scrapers** | Selenium (KenPom, BartTorvik) | requests + urllib (no Selenium) | Moderate |
| **Data leakage** | BartTorvik uses post-tournament CSV | Pre-tournament Time Machine snapshots | **Critical** |
| **Cross-validation** | Random train/test split | Temporal CV (leave-year-out) | **Critical** |
| **Bracket sim** | 10K Monte Carlo (basic) | Forward sim + Monte Carlo + deterministic | Low |
| **Survivor pool** | None | Greedy + optimal + Monte Carlo + live | New feature |
| **Injury adjustments** | None | Full pipeline (Sports Ref + ESPN + LLM) | New feature |

---

## Documentation Index

| Doc | File | What It Covers |
|-----|------|---------------|
| Features & Models | [sports-quant-features-models.md](sports-quant-features-models.md) | Difference features, matchup interactions, symmetrization, FeatureLookup, BartTorvik stats |
| Scraping & Data | [sports-quant-scraping-data-sources.md](sports-quant-scraping-data-sources.md) | Web Archive KenPom, Time Machine BartTorvik, ESPN injuries, Sports Reference, data leakage prevention |
| Calibration & Tuning | [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) | Isotonic calibration, Optuna tuning, debiasing, meta-learner, temporal CV, upset analysis |
| Simulation & Survivor | [sports-quant-simulation-survivor.md](sports-quant-simulation-survivor.md) | Forward sim, Monte Carlo, deterministic mode, survivor pool (greedy/optimal/live) |
| Injury Adjustment | [sports-quant-injury-adjustment.md](sports-quant-injury-adjustment.md) | Player importance scoring, injury scraping, stat degradation, LLM parsing |

---

## Phased Implementation Plan

### Phase 1 — Foundation (Highest ROI)

**Fix data leakage + add difference features + enable temporal CV.**  
These three changes alone could improve accuracy by 5-10 percentage points.

#### 1a. BartTorvik Time Machine API (eliminates data leakage)

**File to modify:** `download_barttorvik.py`  
**Doc reference:** [sports-quant-scraping-data-sources.md](sports-quant-scraping-data-sources.md) §2

Our current `download_barttorvik.py` uses Selenium to download a season-end CSV that includes tournament results. The Time Machine API provides pre-tournament snapshots with no Selenium dependency.

```python
# Replace Selenium download with Time Machine JSON API
# See sports-quant-scraping-data-sources.md for full implementation

import urllib.request, json

def download_barttorvik_snapshot(year: int, date: str = None) -> pd.DataFrame:
    """Download pre-tournament BartTorvik ratings via Time Machine API.
    
    Args:
        year: Season year (e.g. 2024 for 2023-24 season)
        date: Snapshot date "YYYYMMDD". Defaults to Selection Sunday.
    """
    if date is None:
        # Use day before Selection Sunday
        selection_sundays = {
            2022: "20220312", 2023: "20230311",
            2024: "20240316", 2025: "20250315",
        }
        date = selection_sundays.get(year, f"{year}0315")
    
    url = f"https://barttorvik.com/getadvstats.php?year={year}&date={date}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read())
    
    columns = [
        "team", "conf", "record", "adjoe", "adjde", "barthag",
        "efg_o", "efg_d", "to_pct", "to_pct_d", "orb_pct",
        "drb_pct", "ftr", "ftr_d", "twopt_pct", "twopt_pct_d",
        "threept_pct", "threept_pct_d", "adj_tempo", "wins_above_bubble"
    ]
    df = pd.DataFrame(data, columns=columns[:len(data[0])] if data else columns)
    df["year"] = year
    return df
```

**Impact:** Eliminates the #1 data integrity problem (future information leaking into training).

#### 1b. Difference Features (from ~11 to ~34 features)

**File to modify:** `feature_engineering.py`  
**Doc reference:** [sports-quant-features-models.md](sports-quant-features-models.md) §1-2

Our `feature_engineering.py` extracts raw stat values. Sports-quant computes **differences** (team1 - team2) which the model learns much better.

```python
# Add to feature_engineering.py

KENPOM_DIFF_COLS = [
    "AdjEM", "AdjO", "AdjD", "AdjT",
    "eFG_Pct", "TO_Pct", "OR_Pct", "FTRate",
]

BARTTORVIK_DIFF_COLS = [
    "adjoe", "adjde", "barthag", "efg_o", "efg_d",
    "to_pct", "to_pct_d", "orb_pct", "drb_pct",
    "ftr", "adj_tempo", "wins_above_bubble",
]

def compute_difference_features(team1_stats: dict, team2_stats: dict) -> dict:
    """Compute team1 - team2 for all stat columns."""
    features = {}
    
    # KenPom differences
    for col in KENPOM_DIFF_COLS:
        v1 = team1_stats.get(col, 0)
        v2 = team2_stats.get(col, 0)
        features[f"diff_{col}"] = v1 - v2
    
    # BartTorvik differences
    for col in BARTTORVIK_DIFF_COLS:
        v1 = team1_stats.get(col, 0)
        v2 = team2_stats.get(col, 0)
        features[f"diff_{col}"] = v1 - v2
    
    # Seed difference (always include)
    features["seed_diff"] = team1_stats.get("seed", 8) - team2_stats.get("seed", 8)
    
    return features
```

#### 1c. Data Symmetrization (doubles training data)

**File to modify:** `model_training.py`  
**Doc reference:** [sports-quant-features-models.md](sports-quant-features-models.md) §4

```python
def symmetrize_training_data(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Double training data by swapping team perspectives.
    
    For each game (Team A vs Team B, label=1 means A won):
    - Keep original row
    - Add swapped row (Team B vs Team A, label=0)
    
    This ensures the model sees each team equally from both perspectives.
    Only negate columns that start with 'diff_'.
    """
    swapped = df.copy()
    swapped[label_col] = 1 - swapped[label_col]
    
    diff_cols = [c for c in df.columns if c.startswith("diff_")]
    for col in diff_cols:
        swapped[col] = -swapped[col]
    
    return pd.concat([df, swapped], ignore_index=True)
```

#### 1d. Temporal Cross-Validation (prevents year leakage)

**File to modify:** `model_training.py`  
**Doc reference:** [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) §5

Replace random train/test split with leave-year-out CV:

```python
from sklearn.model_selection import LeaveOneGroupOut

def temporal_cross_validate(X, y, years, model_fn):
    """Leave-one-year-out cross-validation.
    
    Args:
        X: Feature matrix
        y: Labels
        years: Array of year per sample (same length as y)
        model_fn: Callable that returns a fitted model
    
    Returns:
        OOF predictions aligned with original indices
    """
    logo = LeaveOneGroupOut()
    oof_preds = np.full(len(y), np.nan)
    
    for train_idx, val_idx in logo.split(X, y, years):
        model = model_fn()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof_preds[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]
    
    return oof_preds
```

---

### Phase 2 — Model Enhancement (High ROI)

**Add probability calibration + Optuna hyperparameter tuning.**  
Expected improvement: 2-5 percentage points in calibration metrics (Brier score, log loss).

#### 2a. Isotonic Calibration

**File to modify:** `model_training.py` (add post-processing)  
**Doc reference:** [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) §1

```python
from sklearn.isotonic import IsotonicRegression

def fit_calibrator(oof_predictions: np.ndarray, oof_labels: np.ndarray):
    """Fit isotonic calibrator on out-of-fold predictions."""
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(oof_predictions, oof_labels)
    return ir

def calibrate(raw_probs: np.ndarray, calibrator) -> np.ndarray:
    """Apply calibration + clip to [0.02, 0.98]."""
    calibrated = calibrator.transform(raw_probs)
    return np.clip(calibrated, 0.02, 0.98)
```

#### 2b. Optuna Hyperparameter Tuning

**File to modify:** `model_training.py`  
**Doc reference:** [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) §2

```python
import optuna

def run_optuna_study(X, y, years, n_trials=200):
    """Bayesian hyperparameter optimization with temporal CV."""
    
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 2000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
        
        oof = temporal_cross_validate(
            X, y, years,
            lambda: XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
        )
        
        valid_mask = ~np.isnan(oof)
        return log_loss(y[valid_mask], oof[valid_mask])
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
```

---

### Phase 3 — Advanced Features (Medium ROI)

**Add matchup interaction features + column-swap debiasing.**

#### 3a. Matchup Interaction Features (+11 features → ~45 total)

**File to modify:** `feature_engineering.py`  
**Doc reference:** [sports-quant-features-models.md](sports-quant-features-models.md) §3

```python
def compute_matchup_features(team1: dict, team2: dict) -> dict:
    """Compute 11 interaction features that capture style matchups."""
    features = {}
    
    # Offense vs defense mismatch (KenPom)
    features["offense_vs_defense_mismatch"] = (
        team1.get("AdjO", 100) - team2.get("AdjD", 100)
    ) - (team2.get("AdjO", 100) - team1.get("AdjD", 100))
    
    # BartTorvik offense vs defense
    features["bart_offense_vs_defense_mismatch"] = (
        team1.get("adjoe", 100) - team2.get("adjde", 100)
    ) - (team2.get("adjoe", 100) - team1.get("adjde", 100))
    
    # Shooting vs opponent defense
    features["shooting_vs_defense"] = (
        team1.get("efg_o", 50) - team2.get("efg_d", 50)
    ) - (team2.get("efg_o", 50) - team1.get("efg_d", 50))
    
    # Tempo mismatch (large diff = uncomfortable for someone)
    features["tempo_mismatch"] = abs(
        team1.get("adj_tempo", 67) - team2.get("adj_tempo", 67)
    )
    
    # Turnover battle
    features["turnover_battle"] = (
        team1.get("to_pct", 18) - team2.get("to_pct_d", 18)
    ) - (team2.get("to_pct", 18) - team1.get("to_pct_d", 18))
    
    # Rebounding battle
    features["rebounding_battle"] = (
        team1.get("orb_pct", 30) - team2.get("drb_pct", 70)
    ) - (team2.get("orb_pct", 30) - team1.get("drb_pct", 70))
    
    # Free throw advantage
    features["ftr_battle"] = (
        team1.get("ftr", 30) - team2.get("ftr_d", 30)
    ) - (team2.get("ftr", 30) - team1.get("ftr_d", 30))
    
    # Seed matchup priors
    s1 = team1.get("seed", 8)
    s2 = team2.get("seed", 8)
    features["seed_diff"] = s1 - s2
    features["higher_seed_is_underdog"] = 1 if s1 > s2 and team1.get("AdjEM", 0) > team2.get("AdjEM", 0) else 0
    
    # Quality agreement: both data sources rank same direction?
    kp_diff = team1.get("AdjEM", 0) - team2.get("AdjEM", 0)
    bt_diff = team1.get("barthag", 0.5) - team2.get("barthag", 0.5)
    features["quality_agreement"] = 1 if (kp_diff > 0) == (bt_diff > 0) else 0
    
    # Combined quality (geometric mean of advantages)
    features["combined_quality"] = (abs(kp_diff) * abs(bt_diff)) ** 0.5 * (1 if kp_diff > 0 else -1)
    
    return features
```

#### 3b. Column-Swap Debiasing

**File to modify:** `predictions.py` or `generate_predictions.py`  
**Doc reference:** [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) §3

```python
def debias_prediction(model, features_a_vs_b: dict, features_b_vs_a: dict) -> float:
    """Average prediction with swapped inputs to remove column ordering bias.
    
    For difference features, b_vs_a = -(a_vs_b), so this is:
        p = (model.predict(a_vs_b) + (1 - model.predict(b_vs_a))) / 2
    """
    p_forward = model.predict_proba(features_a_vs_b)[:, 1]
    p_reverse = model.predict_proba(features_b_vs_a)[:, 1]
    return (p_forward + (1 - p_reverse)) / 2
```

---

### Phase 4 — Meta-Learner Stacking (Medium ROI)

**File to create:** `meta_learner.py`  
**Doc reference:** [sports-quant-calibration-tuning.md](sports-quant-calibration-tuning.md) §4

```python
from sklearn.linear_model import LogisticRegression

def train_meta_learner(base_models, X, y, years):
    """Train stacking meta-learner on OOF predictions from base models.
    
    1. Collect OOF predictions from each base model
    2. Stack into [n_samples, n_models] matrix
    3. Train LogisticRegression on stacked predictions
    """
    oof_stack = np.column_stack([
        temporal_cross_validate(X, y, years, model_fn)
        for model_fn in base_models
    ])
    
    valid = ~np.isnan(oof_stack).any(axis=1)
    meta = LogisticRegression(C=1.0, max_iter=1000)
    meta.fit(oof_stack[valid], y[valid])
    
    return meta, oof_stack
```

---

### Phase 5 — KenPom Scraping Upgrade (Low urgency, high convenience)

**File to modify:** `download_kenpom.py`  
**Doc reference:** [sports-quant-scraping-data-sources.md](sports-quant-scraping-data-sources.md) §1

Replace Selenium with Web Archive + requests approach. Not urgent since KenPom data still works, but eliminates Selenium dependency.

---

### Phase 6 — Simulation Enhancements (Low urgency)

**File to modify:** `bracket_simulation.py`  
**Doc reference:** [sports-quant-simulation-survivor.md](sports-quant-simulation-survivor.md)

Add forward simulation with debiasing, survivor pool optimizer. Our existing Monte Carlo sim works for bracket predictions; the survivor pool is an entirely new use case.

---

### Phase 7 — Injury Adjustment System (Long-term)

**File to create:** `injury_adjustment.py`, `player_importance.py`  
**Doc reference:** [sports-quant-injury-adjustment.md](sports-quant-injury-adjustment.md)

Full player-level injury impact system. Requires two new scrapers and complex name matching. Implement only after Phases 1-3 are stable.

---

## Dependency Graph

```
Phase 1a (Time Machine API)
    └──→ Phase 1b (Difference Features)
              └──→ Phase 1c (Symmetrization)
                        └──→ Phase 1d (Temporal CV)
                                  ├──→ Phase 2a (Calibration)
                                  ├──→ Phase 2b (Optuna Tuning)
                                  └──→ Phase 3a (Matchup Features)
                                            └──→ Phase 3b (Debiasing)
                                                      └──→ Phase 4 (Meta-Learner)
                                                                └──→ Phase 6 (Simulation)

Phase 5 (KenPom scraping) ← Independent, any time
Phase 7 (Injury system) ← Independent, requires Phase 1-3 stable
```

---

## Expected Impact

| Phase | Estimated Accuracy Gain | Effort | Priority |
|-------|------------------------|--------|----------|
| 1a Time Machine | Fixes data leakage (critical correctness) | Low | **P0** |
| 1b Difference features | +3-5% accuracy | Low | **P0** |
| 1c Symmetrization | +1-2% accuracy (more training data) | Low | **P0** |
| 1d Temporal CV | Fixes evaluation bias (critical correctness) | Low | **P0** |
| 2a Calibration | +1-3% Brier score | Low | **P1** |
| 2b Optuna Tuning | +1-3% accuracy | Medium | **P1** |
| 3a Matchup features | +2-4% accuracy | Medium | **P1** |
| 3b Debiasing | +0.5-1% accuracy | Low | **P2** |
| 4 Meta-learner | +1-2% accuracy | Medium | **P2** |
| 5 KenPom scraping | No accuracy change (infra improvement) | Medium | **P3** |
| 6 Simulation | Better bracket picks, new survivor feature | Medium | **P3** |
| 7 Injury system | +1-3% for affected games | High | **P3** |

---

## Files to Modify (Summary)

| Our File | Phase | Changes |
|----------|-------|---------|
| `download_barttorvik.py` | 1a | Replace Selenium with Time Machine API |
| `feature_engineering.py` | 1b, 3a | Add difference features + matchup features |
| `model_training.py` | 1c, 1d, 2a, 2b, 4 | Add symmetrization, temporal CV, calibration, Optuna, meta-learner |
| `generate_predictions.py` | 3b | Add column-swap debiasing |
| `download_kenpom.py` | 5 | Replace Selenium with Web Archive approach |
| `bracket_simulation.py` | 6 | Add forward sim + debiasing + survivor pool |
| `injury_adjustment.py` (new) | 7 | Injury adjustment pipeline |
| `player_importance.py` (new) | 7 | Player scoring system |

---

## Quick Start: Phase 1 Implementation Checklist

- [ ] Update `download_barttorvik.py` with Time Machine API
- [ ] Regenerate historical BartTorvik data using pre-tournament dates
- [ ] Add `compute_difference_features()` to `feature_engineering.py`
- [ ] Add `compute_matchup_features()` to `feature_engineering.py`
- [ ] Add `symmetrize_training_data()` to `model_training.py`
- [ ] Replace random split with temporal CV in `model_training.py`
- [ ] Retrain models and compare accuracy
- [ ] Update Streamlit pages to use new features
