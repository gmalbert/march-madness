# Roadmap: Reducing Spread & Total MAE

> **Current State** (March 2026)
> | Metric | Baseline (3 feat) | Current (11 feat) | Vegas Line MAE |
> |--------|-------------------|-------------------|----------------|
> | Spread MAE | 11.61 | **10.64** | ~8.5–9.0* |
> | Total MAE | 16.00 | **15.72** | ~9.5–10.5* |
> | Moneyline Acc | 65.3% | 68.0% | ~73–75%* |
>
> \* Industry consensus for closing-line accuracy on college basketball.

---

## 0. Is MAE the Right Metric?

**Yes — MAE is the correct primary metric for spread and total prediction.** Here's why:

| Property | MAE | RMSE | MAPE |
|----------|-----|------|------|
| Interpretability | ✅ "Off by X points on average" | Harder — penalizes outliers more | % error not intuitive for scores near 0 |
| Robustness to outliers | ✅ Not distorted by blowouts | ❌ One 40-pt miss dominates RMSE | N/A |
| Matches betting use case | ✅ 1-pt error = 1-pt error | No — 10-pt error counts as 100 | Not relevant |
| Used by industry | ✅ Standard for ATS/total evaluation | Secondary metric | Rarely used |

**Secondary metrics to track alongside MAE:**
- **Median Absolute Error (MdAE)**: Ignores outliers entirely; gives "typical" accuracy.
- **Cover Rate / ATS Accuracy**: What % of the time does the model correctly pick ATS (against the spread)? This is the _betting_ accuracy. An MAE of 10.64 can still win bets if errors are symmetric around the line.
- **Calibration (predicted vs. actual)**: Plot predicted spread vs. actual spread; the slope should be ~1.0 and intercept ~0. If the model is systematically conservative (predictions cluster near 0), calibration fixes alone can drop MAE.
- **Over/Under hit rate**: Same concept for totals.

**Key insight**: Closing Vegas lines achieve ~8.5–9.0 MAE on spreads. Your 10.64 is ~1.6–2.1 points away. This is the gap to close. For totals, Vegas is ~9.5–10.5 MAE — your 15.72 is ~5+ points away, representing a bigger opportunity.

---

## 1. Quick Wins (Est. Impact: 1–3 pts MAE reduction)

### 1.1 Use Vegas Lines as a Feature (Not Just a Target)
**The single highest-impact change.**

Your training data already contains `betting_spread` and `betting_over_under`, but these are **not included as features** — only as comparison targets. Vegas lines encode massive amounts of information (injuries, matchups, public betting patterns, situational factors) that your model doesn't capture.

**Implementation:**
```python
# In get_feature_columns() in train_tournament_models.py
def get_feature_columns(df: pd.DataFrame, model_type: str) -> List[str]:
    if model_type in ("spread", "moneyline"):
        cols = [c for c in df.columns if c.startswith("spread_")]
    elif model_type == "total":
        cols = [c for c in df.columns if c.startswith("total_")]

    # Add enriched KenPom/BartTorvik features
    enriched = [c for c in df.columns if c.startswith("kenpom_") or c.startswith("bart_")]
    cols += enriched

    # NEW: Add Vegas line features
    if model_type == "spread" and "betting_spread" in df.columns:
        cols.append("betting_spread")
    if model_type == "total" and "betting_over_under" in df.columns:
        cols.append("betting_over_under")
    # Moneylines encode implied win probability
    if "home_moneyline" in df.columns and "away_moneyline" in df.columns:
        cols += ["home_moneyline", "away_moneyline"]

    return cols
```

**Why this works:** If you regress actual_spread ~ betting_spread + model_features, the model learns to _adjust_ the line rather than predict from scratch. This is how professional models work — they don't try to beat Vegas outright, they find where Vegas is slightly wrong.

**Expected impact:** 1.5–3.0 MAE reduction on spreads, 2–4 on totals.

**Caveat:** Only ~62% of your training rows have betting lines. Handle NaN by either:
- Training on the subset with lines (smaller dataset but much better signal)
- Imputing missing lines with model predictions (two-stage approach)

---

### 1.2 Fix the Ensemble (Weighted Average, Not Simple Mean)
Currently in `generate_predictions.py`, the ensemble averages all model variants with equal weight:
```python
np.mean()  # line 179
```

Replace with learned weights:

```python
# Optimal ensemble weights via leave-one-out or validation set
# After training all models, do a simple stacking regression:
from sklearn.linear_model import Ridge

# On validation set, collect predictions from each model
# Then fit: actual = w1 * pred_ridge + w2 * pred_xgb + w3 * pred_rf
stacker = Ridge(alpha=1.0)
stacker.fit(np.column_stack([pred_ridge, pred_xgb, pred_rf]), y_val)
```

A stacked ensemble (meta-learner on top of base model predictions) typically gives 0.2–0.5 MAE improvement over simple averaging.

---

### 1.3 Expand Spread Feature Set (Use `total_*` Features for Spread Too)
Currently, spread models only see `spread_*` features and enriched KenPom/BartTorvik features. But `total_*` features (pace, combined offensive efficiency) **contain information about spread** — a high-pace game with one strong offense and one weak one will have a different spread dynamic than a slow grind.

```python
# In get_feature_columns(), change:
def get_feature_columns(df: pd.DataFrame, model_type: str) -> List[str]:
    # For spread/moneyline: use BOTH spread_* and total_* features
    if model_type in ("spread", "moneyline"):
        cols = [c for c in df.columns if c.startswith("spread_") or c.startswith("total_")]
    elif model_type == "total":
        cols = [c for c in df.columns if c.startswith("total_") or c.startswith("spread_")]
    
    enriched = [c for c in df.columns if c.startswith("kenpom_") or c.startswith("bart_")]
    cols += enriched
    return cols
```

**Expected impact:** 0.3–0.8 MAE improvement.

---

### 1.4 Hyperparameter Tuning (Beyond `RandomizedSearchCV`)
Current XGBoost config: `n_estimators=200, max_depth=4, learning_rate=0.05`. The `model_analysis.py` grid has a narrow search space.

Use **Optuna** for Bayesian hyperparameter optimization:

```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'gamma': trial.suggest_float('gamma', 1e-8, 5.0, log=True),
    }
    model = XGBRegressor(**params, random_state=42)
    # Use leave-one-year-out CV for proper tournament evaluation
    scores = []
    for year in years:
        # ... train on all other years, test on `year` tournament games
        scores.append(mae)
    return np.mean(scores)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=200)
```

Add `optuna` to requirements:
```
optuna>=3.0
```

**Expected impact:** 0.3–1.0 MAE improvement.

---

## 2. Feature Engineering (Est. Impact: 1–3 pts MAE reduction)

### 2.1 Recency-Weighted Stats (Last 10 / Last 5 / Last 3 Games)
Season averages mask hot/cold streaks. Teams entering the tournament on a 10-game win streak behave differently than teams that limped in.

**New features to add:**
```python
# In feature_engineering.py
def calculate_recency_features(team_recent_games: list) -> dict:
    """Calculate performance over last N games."""
    features = {}
    for window in [3, 5, 10]:
        recent = team_recent_games[-window:]
        features[f'last_{window}_margin'] = np.mean([g['margin'] for g in recent])
        features[f'last_{window}_off_eff'] = np.mean([g['off_eff'] for g in recent])
        features[f'last_{window}_def_eff'] = np.mean([g['def_eff'] for g in recent])
        features[f'last_{window}_win_pct'] = np.mean([g['won'] for g in recent])
    return features
```

**Data source:** CBBD API already has game-level data; aggregate it pre-tournament.

### 2.2 Seed-Based Features (for Tournament Only)
Seed matchup patterns are highly predictive in the tournament. 1-vs-16 behaves differently than 5-vs-12.

```python
def calculate_seed_features(home_seed: int, away_seed: int) -> dict:
    return {
        'seed_diff': home_seed - away_seed,
        'seed_sum': home_seed + away_seed,
        'higher_seed': min(home_seed, away_seed),
        'lower_seed': max(home_seed, away_seed),
        'seed_product': home_seed * away_seed,  # Nonlinear: 1v16=16, 8v9=72
        'is_chalk': int(home_seed < away_seed),  # Favorite indicator
    }
```

### 2.3 Tournament Round Feature
Later rounds are closer. The average margin decreases round by round:
- Round of 64: ~10.5 pts
- Round of 32: ~9.2 pts
- Sweet 16: ~7.8 pts
- Elite 8: ~6.5 pts
- Final Four: ~5.8 pts

```python
# Round number as a feature helps the model learn that later rounds → tighter games
'tournament_round': round_number  # 1=R64, 2=R32, ..., 6=Championship
```

### 2.4 Home Court / Neutral Site Features
Tournament games are on neutral courts, but some teams travel less distance to venue. Geography matters.

```python
def calculate_location_features(team_lat, team_lon, venue_lat, venue_lon):
    return {
        'travel_distance': haversine(team_lat, team_lon, venue_lat, venue_lon),
        'is_neutral': 1,  # all tournament games
        'distance_advantage': travel_dist_away - travel_dist_home,
    }
```

### 2.5 Opponent-Adjusted Shooting Metrics
Current features use raw FG%, eFG%, etc. Adjusted metrics (opponent-adjusted) are more predictive:

```python
# Instead of raw efg_pct, use:
'adj_efg_diff': home_adj_efg - away_adj_efg  # eFG% adjusted for opponent defense
'adj_to_rate_diff': ...  # Turnover rate adjusted for opponent forcing ability
```

KenPom and BartTorvik both provide these — pull them explicitly rather than using the aggregate ratings only.

---

## 3. New Data Sources (Est. Impact: 1–2 pts MAE reduction)

### 3.1 Evan Miya Ratings
[evanmiya.com](https://evanmiya.com) — Player-level Bayesian ratings. Team-level aggregates from player-level models capture roster changes, injuries, and individual matchups that team-level stats miss.

### 3.2 Haslametrics
[haslametrics.com](https://haslametrics.com) — Another independent efficiency system. Combining 3+ independent rating systems (KenPom, BartTorvik, Haslametrics) reduces systematic bias in any single system.

### 3.3 ESPN BPI / Sagarin / Massey
These composite ratings provide independent signals:
- **ESPN BPI** (Basketball Power Index): Available via ESPN API
- **Sagarin**: Pure mathematical model, no human judgment
- **Massey Composite**: Aggregates 50+ computer ratings

### 3.4 Player-Level Data
**Biggest missing dimension.** When a star player is injured or in foul trouble, team-level stats are meaningless.

- **Injury reports** (e.g., DonBest, CBS Sports)
- **Player efficiency ratings (PER)**: Aggregate offensive/defensive contributions
- **Minutes-weighted player importance**: How much does the outcome swing if player X sits?

### 3.5 Betting Market Features
You already capture opening lines. Add:
- **Line movement** (opening → closing): Sharp line movement indicates where the smart money is
- **Public betting percentages**: Available from Action Network, Pregame.com
- **Reverse line movement**: Line moves opposite to public betting → sharp money signal
- **Steam moves**: Sudden coordinated line movements across sportsbooks

```python
def calculate_betting_market_features(opening_line, current_line, public_pct):
    return {
        'line_movement': current_line - opening_line,
        'public_pct_home': public_pct,
        'reverse_line_move': int(
            (current_line > opening_line and public_pct > 0.5) or
            (current_line < opening_line and public_pct < 0.5)
        ),
    }
```

### 3.6 Tempo-Free Shooting Splits
Break down shooting beyond eFG%:
- Rim FG%, mid-range FG%, 3PT% (by zone)
- Free throw rate and FT%
- Shot distribution (% of shots at rim vs. 3pt vs. mid-range)

Available from **Bart Torvik** and **hoop-math.com**.

---

## 4. Model Architecture Improvements (Est. Impact: 0.5–2 pts MAE)

### 4.1 Stacked Ensemble (Meta-Learner)
Replace simple averaging with a two-level model:

```
Level 0 (Base models):
  - XGBoost
  - Random Forest  
  - Ridge Regression
  - LightGBM (add this)
  - CatBoost (add this)

Level 1 (Meta-learner):
  - Ridge regression on Level-0 out-of-fold predictions
  - Input: [pred_xgb, pred_rf, pred_ridge, pred_lgbm, pred_catboost]
  - Output: final prediction
```

```python
from sklearn.model_selection import KFold
import lightgbm as lgb
from catboost import CatBoostRegressor

def stacked_ensemble(X, y, weights):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_preds = np.zeros((len(X), 5))  # 5 base models
    
    models = [
        XGBRegressor(n_estimators=300, max_depth=5, learning_rate=0.05),
        RandomForestRegressor(n_estimators=300, max_depth=8),
        Ridge(alpha=1.0),
        lgb.LGBMRegressor(n_estimators=300, max_depth=5, learning_rate=0.05),
        CatBoostRegressor(iterations=300, depth=5, learning_rate=0.05, verbose=0),
    ]
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
        for model_idx, model in enumerate(models):
            model.fit(X.iloc[train_idx], y.iloc[train_idx],
                     sample_weight=weights.iloc[train_idx])
            oof_preds[val_idx, model_idx] = model.predict(X.iloc[val_idx])
    
    # Meta-learner
    meta = Ridge(alpha=1.0)
    meta.fit(oof_preds, y)
    return meta, models
```

Add to requirements:
```
lightgbm>=4.0
catboost>=1.2
```

### 4.2 Quantile Regression for Better Uncertainty
Instead of predicting only the mean, predict the 10th, 50th, and 90th percentiles:

```python
from sklearn.ensemble import GradientBoostingRegressor

# Predict median (more robust than mean for MAE)
model_median = GradientBoostingRegressor(loss='quantile', alpha=0.5)
model_q10 = GradientBoostingRegressor(loss='quantile', alpha=0.1)
model_q90 = GradientBoostingRegressor(loss='quantile', alpha=0.9)
```

**Key insight:** MAE is minimized by the **median**, not the mean. If your models optimize for MSE (squared error) but you evaluate with MAE, there's an inherent mismatch. Train XGBoost with `objective='reg:absoluteerror'` instead of `objective='reg:squarederror'`:

```python
# Current (optimizes for MSE):
xgb = XGBRegressor(objective='reg:squarederror', ...)

# Better for MAE (optimizes MAE directly):
xgb = XGBRegressor(objective='reg:absoluteerror', ...)
```

This is a **free improvement** — your evaluation metric should match your training objective.

### 4.3 Neural Network Approach (Low Priority, High Ceiling)
A simple feedforward neural network with embeddings for team IDs can capture nonlinear matchup-specific patterns:

```python
import torch
import torch.nn as nn

class SpreadNet(nn.Module):
    def __init__(self, n_teams=400, emb_dim=16, n_features=20):
        super().__init__()
        self.team_emb = nn.Embedding(n_teams, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(2 * emb_dim + n_features, 128),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, home_id, away_id, features):
        h = self.team_emb(home_id)
        a = self.team_emb(away_id)
        x = torch.cat([h, a, features], dim=1)
        return self.net(x)
```

This learns each team's latent "style" and how it interacts with opponents — but requires more data and careful regularization.

---

## 5. Training Methodology (Est. Impact: 0.5–1.5 pts MAE)

### 5.1 Fix the Loss Function Mismatch ⚠️ HIGH PRIORITY
**This is likely the single most impactful quick fix.**

Your models train with `reg:squarederror` (optimizes RMSE) but evaluate with MAE. These objectives have different optima. Switch to:

```python
XGBRegressor(objective='reg:absoluteerror', ...)
```

Or use Huber loss for a compromise:
```python
XGBRegressor(objective='reg:pseudohubererror', ...)
```

### 5.2 Proper Cross-Validation Strategy
Current: `train_test_split(test_size=0.2, random_state=42)` — a single random split.

Problems:
- **Temporal leakage**: 2024 training data predicts 2018 test games (future → past)
- **Tournament contamination**: Regular-season games from the same season as tournament test games leak seasonal patterns

Better approach — **expanding-window temporal CV**:
```python
def temporal_cv(df, years):
    """Train on years [2016..Y-1], test on year Y tournament games."""
    results = []
    for test_year in years:
        train = df[df['season'] < test_year]
        test = df[(df['season'] == test_year) & (df['game_type'] == 'tournament')]
        # Train model, evaluate on test
        results.append(evaluate(train, test))
    return results
```

Your `lovo_cv_tournament()` already does leave-one-year-out — **use those MAE numbers as the true performance metric**, not the 80/20 random split which is artificially optimistic.

### 5.3 Tournament Weighting Tuning
Currently fixed at 3× or 5×. Optimize this:

```python
for weight in [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 15.0]:
    # Train with this weight, evaluate via LOYO CV
    mae = loyo_cv(df, tournament_weight=weight)
    print(f"Weight {weight}: MAE = {mae}")
```

### 5.4 Regularization Review
Current Random Forest: `max_depth=6, min_samples_split=10` — reasonable but potentially too constrained. XGBoost `max_depth=4` may be too shallow for 18+ features.

---

## 6. Data Quality & Pipeline Fixes (Est. Impact: 0.3–1.0 pts MAE)

### 6.1 Handle Missing KenPom/BartTorvik Data
Currently: missing enriched features filled with 0. This is problematic — 0 means "no difference," which is a strong false assertion.

Better approaches:
- **Train on complete cases only** (62% of data → ~16k games) — loses data but no imputation noise
- **Use median imputation** from the training set
- **Use a missing indicator**: Add binary features `kenpom_available`, `bart_available`
- **Multiple imputation** via `sklearn.impute.IterativeImputer`

### 6.2 Projected Total Formula Bug
In `feature_engineering.py`, the projected total formula looks off:
```python
projected_total = (avg_off_eff + avg_def_eff) / 2 * (avg_tempo / 100) * 0.8
```
This should be something like:
```python
# Each team's expected points = (own OE * opp avg possessions)
# Total ≈ team1_off_eff * tempo/100 + team2_off_eff * tempo/100
# Simplified: combined_off_eff * avg_tempo / 100
projected_total = (home_off_eff * avg_tempo / 100) + (away_off_eff * avg_tempo / 100)
```

### 6.3 Feature Scaling Inconsistency
Ridge/Linear models use `StandardScaler`, XGBoost/RF do not. This is correct behavior — but ensure the scaler is always saved and loaded correctly in the prediction pipeline.

---

## 7. Realistic MAE Targets

| Target | Spread MAE | Total MAE | Difficulty |
|--------|-----------|-----------|------------|
| **Current** | 10.64 | 15.72 | — |
| **Phase 1** (Quick wins: §1 + §5.1) | 8.5–9.5 | 12.0–13.5 | Moderate |
| **Phase 2** (Features: §2 + §3) | 7.8–8.5 | 11.0–12.0 | Significant |
| **Phase 3** (Architecture: §4) | 7.5–8.2 | 10.5–11.5 | Hard |
| **Theoretical floor** | ~7.0–7.5 | ~9.0–10.0 | Near-impossible |

**Why a floor exists:** College basketball has irreducible variance — injuries during games, hot/cold shooting streaks, referee variance, and random bounces create noise no model can predict. Vegas closing lines represent ~$100B+ of collective intelligence, and even they can't do better than ~8.5 MAE on spreads.

---

## 8. Prioritized Action Plan

### Phase 1: Training Fixes (Do These First)
- [ ] **§5.1** Switch XGBoost `objective` from `reg:squarederror` → `reg:absoluteerror`
- [ ] **§1.1** Add `betting_spread` and `betting_over_under` as input features
- [ ] **§1.3** Cross-pollinate spread/total feature sets
- [ ] **§5.2** Switch primary evaluation to LOYO CV (tournament holdout)
- [ ] **§6.1** Handle missing enriched features properly (median imputation + indicator)
- [ ] **§6.2** Fix projected total formula

### Phase 2: Feature Expansion
- [ ] **§2.1** Add recency features (last 3/5/10 games)
- [ ] **§2.2** Add seed-based features for tournament games
- [ ] **§2.3** Add tournament round feature
- [ ] **§1.4** Run Optuna hyperparameter search (add to requirements)
- [ ] **§1.2** Implement stacking ensemble (replace simple mean)

### Phase 3: New Data & Models
- [ ] **§3.1–3.3** Integrate additional rating systems (Evan Miya, Haslametrics, BPI)
- [ ] **§3.5** Add betting market features (line movement, public %)
- [ ] **§4.1** Full stacked ensemble with LightGBM + CatBoost
- [ ] **§4.2** Quantile regression for uncertainty
- [ ] **§3.4** Player-level injury impact features

### Phase 4: Advanced
- [ ] **§4.3** Neural network with team embeddings
- [ ] **§2.4** Geographic travel-distance features
- [ ] **§2.5** Opponent-adjusted shooting splits
- [ ] **§3.6** Tempo-free shooting zone data

---

## 9. Other Suggestions You May Have Missed

### 9.1 Calibration Plot
Before chasing lower MAE, check whether your model is **biased**:
```python
import matplotlib.pyplot as plt

# Plot predicted vs actual spread
plt.scatter(y_pred, y_actual, alpha=0.1)
plt.plot([-30, 30], [-30, 30], 'r--')  # Perfect calibration line
plt.xlabel('Predicted Spread')
plt.ylabel('Actual Spread')
```
If the slope ≠ 1.0 or intercept ≠ 0, a simple linear recalibration (`actual = a * predicted + b`) can improve MAE for free.

### 9.2 Prediction Shrinkage
Models often predict extreme spreads that rarely materialize. Shrinking predictions toward 0 by 10–20% can reduce MAE:
```python
shrink_factor = 0.85
adjusted_pred = predicted_spread * shrink_factor
```
This works because college basketball spreads have a fat-tailed distribution — extreme outcomes are rarer than models expect.

### 9.3 Separate Models for Different Game Types
- **Power conference vs. mid-major** games have different dynamics
- **Tournament games specifically** (you partly do this with tournament weighting, but dedicated models may help more)
- **First round vs. later rounds** (bigger upsets in first round)

### 9.4 Feature Importance Pruning
Run SHAP analysis to find which features the model actually uses:
```python
import shap
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values, X_test)
```
Remove features with near-zero importance — they add noise without signal.

### 9.5 Watchlist for Overfitting
Your comparison shows XGBoost tournament MAE of 7.633 on _all tournament games_ (which includes training data) vs. 9.2–10.6 on LOYO holdouts. That's a **2+ point overfit gap**. The LOYO numbers are the real metric. Any improvement should be validated via LOYO CV, not the in-sample comparison.

### 9.6 Early Stopping for XGBoost
```python
xgb = XGBRegressor(n_estimators=1000, early_stopping_rounds=50, ...)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)])
```
This automatically finds the optimal number of trees instead of fixing at 200.

---

## 10. Requirements Changes

Add these to `requirements.txt` for the full roadmap:
```
optuna>=3.0
lightgbm>=4.0
catboost>=1.2
shap>=0.42
```

Optional for neural network approach:
```
torch>=2.0
```
