# Roadmap: Matchup Interaction Features

> **Priority**: P1 — High impact once difference features are in place  
> **Estimated Effort**: Medium  
> **Source**: sports-quant `_feature_builder.py` matchup feature groups  
> **Impact**: Captures cross-source and style-matchup signals that pure stat diffs miss

---

## Problem Statement

Our current features are pure stat differentials — "Team A's offense is X points better than Team B's defense." But tournament games are **matchup-dependent**: a high-tempo team playing against a slow, methodical team creates a dynamic that raw diffs don't capture. Sports-quant's kernel analysis found 11 matchup interaction features that add signal beyond stat differences.

---

## 11 Matchup Interaction Features

### Group 1: Offensive vs Defensive Style Interactions (4 features)

These capture how one team's strength lines up against the other's weakness:

```
offense_vs_defense_mismatch:
  (Team1_AdjO - Team2_AdjD) - (Team2_AdjO - Team1_AdjD)
  → Net offensive mismatch from KenPom

bart_offense_vs_defense_mismatch:
  (Team1_BartAdjOE - Team2_BartAdjDE) - (Team2_BartAdjOE - Team1_BartAdjDE)
  → Same from BartTorvik (independent confirmation)

offense_defense_product:
  adjO_diff × adjD_diff
  → Captures "both sides of the ball" advantage magnitude
  → Even symmetry (invariant under team swap)

bart_offense_defense_product:
  bart_adjOE_diff × bart_adjDE_diff
  → BartTorvik version of the same
```

### Group 2: Tempo Mismatch (3 features)

Tempo mismatches matter enormously in March Madness — slow teams forced to play fast make mistakes:

```
tempo_mismatch_magnitude:
  |adjT_diff|
  → Absolute size of tempo gap (even symmetry)

tempo_x_quality_interaction:
  adjT_diff × adjEM_diff
  → Tempo advantage amplified by overall quality gap

tempo_x_seed_interaction:
  adjT_diff × seed_diff
  → Tempo advantage amplified by expected matchup closeness
```

### Group 3: Historical Seed Priors (2 features)

Empirical seed-vs-seed upset rates as a Bayesian prior:

```
seed_upset_prior_centered:
  Historical P(Team1 wins) - 0.5 based on seed matchup
  → e.g., 5-vs-12 → centered prior of +0.15 for the 5-seed

  Hardcoded rates (2003-2024):
    (1,16): 0.01, (2,15): 0.06, (3,14): 0.15, (4,13): 0.20
    (5,12): 0.35, (6,11): 0.37, (7,10): 0.39, (8,9): 0.48

seed_x_quality_gap:
  seed_upset_prior × adjEM_diff
  → When the historical prior disagrees with the quality gap, something interesting is happening
```

### Group 4: Quality Consistency (2 features)

Agreement between independent rating systems as a confidence signal:

```
quality_source_agreement:
  adjEM_diff × bart_barthag_diff
  → If KenPom and BartTorvik agree on who's better → high confidence
  → Disagreement → high uncertainty (valuable for upset detection)

sos_quality_interaction:
  sos_adjEM_diff × adjEM_diff
  → Whether the better team earned their rating against tough competition
```

---

## Symmetry Properties

Important for correct debiasing — some features are **odd** (negate under swap) and some are **even** (invariant):

| Feature | Symmetry | Under Team Swap |
|---------|----------|-----------------|
| offense_vs_defense_mismatch | Odd | Negate |
| bart_offense_vs_defense_mismatch | Odd | Negate |
| offense_defense_product | **Even** | Keep |
| bart_offense_defense_product | **Even** | Keep |
| tempo_mismatch_magnitude | **Even** | Keep |
| tempo_x_quality_interaction | **Even** | Keep |
| tempo_x_seed_interaction | **Even** | Keep |
| seed_upset_prior_centered | Odd | Negate |
| seed_x_quality_gap | **Even** | Keep |
| quality_source_agreement | **Even** | Keep |
| sos_quality_interaction | **Even** | Keep |

The `symmetrize_training_data()` function must negate **odd** features and keep **even** features. Sports-quant defines `MATCHUP_EVEN_FEATURES` as a frozenset for this.

---

## Implementation Plan

### Step 1: Create `matchup_features.py`

- Define `SEED_MATCHUP_PRIORS` dict
- `seed_upset_prior(seed1, seed2) -> float`
- `compute_matchup_features(team1_stats, team2_stats, seed1, seed2) -> dict`

### Step 2: Integrate with Feature Pipeline

- Add to `compute_all_difference_features()` from difference-features roadmap
- Results in 25 base diffs + 11 matchup = **36 total combined features**

### Step 3: Update Symmetrization

- Track which features are even vs odd
- Even features stay unchanged during symmetrization
- Odd features get negated

### Step 4: Validate Feature Importance

After training:
- Check that matchup features appear in top-20 feature importance
- Features with zero importance should be dropped for parsimony

---

## Acceptance Criteria

- [ ] All 11 matchup features computed correctly
- [ ] Symmetry properties verified via unit tests
- [ ] Feature importance confirms at least some matchup features add signal
- [ ] Backtest accuracy improves over difference-features-only baseline

---

## Dependencies

- **Difference features** (`roadmap-difference-features-and-debiasing.md`) — matchup features build on top
- **Both KenPom and BartTorvik data** — needed for cross-source features (Groups 1, 4)
