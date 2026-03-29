# Roadmap: Project Architecture & Code Organization

> **Priority**: P1 — Reduces friction for every future improvement  
> **Estimated Effort**: Medium  
> **Not in sports-quant analysis** — identified from reviewing the current repo structure  
> **Impact**: Faster iteration, fewer bugs, easier testing

---

## Problem Statement

The current repo has **42+ Python files** at the root level, many with overlapping responsibilities. There are 3+ different model training files (`model_training.py`, `advanced_model_training.py`, `train_tournament_models.py`, `betting_models.py`), multiple feature engineering approaches scattered across files, and 60+ scripts in `scripts/`. This makes it hard to:

1. Know which file is the "source of truth" for any operation
2. Avoid breaking changes when modifying shared code
3. Test individual components
4. Onboard or return to the project after a break

Sports-quant uses a clean package structure (`src/sports_quant/march_madness/`) with single-responsibility modules. We should move toward a similar organization.

---

## Proposed Structure

```
march-madness/
├── model_config.yaml              # Centralized configuration
├── predictions.py                 # Streamlit app entry point (keep at root)
├── pages/                         # Streamlit pages (keep as-is)
│
├── src/                           # Core package
│   ├── __init__.py
│   ├── config.py                  # Load model_config.yaml, paths, env vars
│   ├── teams.py                   # Team name canonicalization
│   │
│   ├── data/                      # Data ingestion
│   │   ├── __init__.py
│   │   ├── kenpom.py              # KenPom scraper
│   │   ├── barttorvik.py          # BartTorvik Time Machine API
│   │   ├── cbbd.py                # College Basketball Data API
│   │   ├── odds.py                # Live odds fetching
│   │   └── loader.py              # Unified data loading (from efficiency_loader.py)
│   │
│   ├── features/                  # Feature engineering
│   │   ├── __init__.py
│   │   ├── difference.py          # Difference features (KenPom + BartTorvik)
│   │   ├── matchup.py             # Matchup interaction features
│   │   ├── spread.py              # Spread prediction features
│   │   ├── total.py               # Total prediction features
│   │   └── builder.py             # Feature pipeline orchestrator
│   │
│   ├── modeling/                   # Training and prediction
│   │   ├── __init__.py
│   │   ├── training.py            # Unified model training
│   │   ├── ensemble.py            # Ensemble methods + meta-learner
│   │   ├── calibration.py         # Probability calibration
│   │   ├── debiasing.py           # Positional debiasing
│   │   └── tuning.py              # Optuna hyperparameter optimization
│   │
│   ├── evaluation/                 # Metrics and backtesting
│   │   ├── __init__.py
│   │   ├── backtest.py            # Temporal backtesting orchestrator
│   │   ├── metrics.py             # Log loss, Brier, ESPN scoring, etc.
│   │   ├── upsets.py              # Upset analysis
│   │   └── plots.py               # Visualization (learning curves, feature importance)
│   │
│   ├── simulation/                 # Bracket simulation
│   │   ├── __init__.py
│   │   ├── bracket.py             # Forward simulation engine
│   │   ├── monte_carlo.py         # Monte Carlo simulation
│   │   └── survivor.py            # Survivor pool optimizer
│   │
│   └── betting/                    # Betting-specific
│       ├── __init__.py
│       ├── value.py               # Value bet detection
│       ├── kelly.py               # Kelly criterion sizing
│       └── roi.py                 # ROI tracking
│
├── scripts/                        # One-off and utility scripts
│   ├── retune.py                  # End-to-end hyperparameter retune
│   ├── match_teams.py             # Team name matching
│   ├── generate_predictions.py    # Generate daily predictions
│   └── ...
│
├── tests/                          # Test suite (moved from scripts/)
│   ├── test_features.py
│   ├── test_training.py
│   ├── test_simulation.py
│   └── ...
│
├── data_files/                     # Data (gitignored except mappings)
├── docs/                           # Documentation
└── pages/                          # Streamlit pages
```

---

## Migration Strategy

Do this **incrementally**, not all at once:

### Phase 1: Create `src/` skeleton with `config.py`
- Create the directory structure
- Move config loading to `src/config.py`
- All other files import from `src.config`

### Phase 2: Consolidate feature engineering
- Merge `features.py` + `feature_engineering.py` → `src/features/`
- Add difference feature computation
- Keep old files as thin wrappers that import from `src/`

### Phase 3: Consolidate model training
- Merge `model_training.py` + `advanced_model_training.py` + `train_tournament_models.py` → `src/modeling/training.py`
- Single training function with config-driven behavior

### Phase 4: Move tests to `tests/`
- Extract test files from `scripts/` to `tests/`
- Use `pytest` (already available via scikit-learn dependency)

### Phase 5: Clean up root
- Move remaining root-level scripts to `scripts/`
- Root should only have: `predictions.py`, config, requirements, README

---

## Key Design Principles (from sports-quant)

### 1. Frozen Dataclasses for Data Structures
```python
@dataclass(frozen=True)
class TeamStats:
    team: str
    year: int
    seed: int
    features: dict[str, float]
```

### 2. Pure Functions Where Possible
Feature computation functions should be stateless — input dataframe, output dataframe. No side effects.

### 3. Single Responsibility
Each file does one thing. `_features.py` defines features. `_feature_builder.py` computes them. `_debiasing.py` handles debiasing. No file does everything.

### 4. Explicit Feature Definitions
All feature names defined as constants (tuples, not lists — immutable). No magic strings scattered across the codebase.

---

## Acceptance Criteria

- [ ] `src/` package exists with clean module structure
- [ ] `config.py` is the single source of truth for paths and parameters
- [ ] No duplicate feature engineering logic
- [ ] No duplicate model training logic
- [ ] Tests live in `tests/` and run via `pytest`
- [ ] Root directory has ≤10 files (down from 42+)
- [ ] All existing functionality preserved (Streamlit app still works)

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Breaking existing imports | Medium | Migrate incrementally, keep thin wrappers |
| Streamlit can't find modules | Low | Add `src/` to sys.path or use proper package install |
| Effort feels unrewarding | Low | Each phase delivers tangible simplification |
