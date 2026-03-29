# Roadmap: Forward Bracket Simulation Engine

> **Priority**: P1 — Makes bracket prediction realistic  
> **Estimated Effort**: Medium-High  
> **Source**: sports-quant `simulate.py`, `_bracket.py`, `_bracket_builder.py`, bracket-simulation-spec.md  
> **Impact**: Replaces oracle-matchup evaluation with cascading bracket sim; enables survival pool optimization

---

## Problem Statement

Our current `bracket_simulation.py` runs Monte Carlo simulations on the bracket. Sports-quant's approach is more sophisticated in several ways:

1. **Forward simulation** — fills the bracket round by round, feeding predicted winners into the next round (so errors cascade realistically)
2. **Three simulation modes** — deterministic, Monte Carlo, and pre-computed pairwise probability lookup
3. **Upset analysis** — tracks which upsets are predicted by what % of ensemble models
4. **Survivor pool optimization** — multiple strategies (greedy, bracket-aware, Monte Carlo optimal)

---

## Part 1: Forward Simulation Design

### Deterministic Mode
- For each game, pick the team with higher predicted win probability
- Fills bracket round-by-round: R64 → R32 → S16 → E8 → F4 → NCG
- Produces a single bracket
- Useful for "best guess" predictions

### Monte Carlo Mode  
- For each game, sample winner according to predicted probability
- Run N simulations (1000+)
- Track how often each team reaches each round
- Pre-compute all 2016 possible pairwise probabilities once, then look up during simulation

### Pre-Computed Pairwise Probabilities
Sports-quant's key optimization:
```python
# Compute probabilities for ALL possible matchups once
probs = {}
for i, team1 in enumerate(all_teams):
    for team2 in all_teams[i+1:]:
        p = predict_matchup(team1, team2)
        probs[(team1.name, team2.name)] = p
        probs[(team2.name, team1.name)] = 1.0 - p

# During MC simulation, just look up:
p = probs[(team1.name, team2.name)]
winner = team1 if rng.random() < p else team2
```

This makes MC simulation extremely fast (no model inference during sampling).

---

## Part 2: Bracket Data Structures

```python
@dataclass
class BracketGame:
    round_name: str       # "R64", "R32", "S16", "E8", "F4", "NCG"
    team1: TeamStats
    team2: TeamStats
    winner: TeamStats
    win_probability: float
    is_upset: bool

@dataclass
class SimulationResult:
    games: list[BracketGame]           # All games in bracket
    champion: TeamStats
    bracket_score: int                 # ESPN-style scoring
    round_results: dict[str, list]     # Per-round results
    
@dataclass
class MonteCarloResult:
    championship_probs: dict[str, float]   # Team → P(champion)
    round_advance_probs: dict[str, dict]   # Team → {round: probability}
    modal_bracket: SimulationResult        # Most common outcome
    n_simulations: int
```

---

## Part 3: Popular Upset Tracking

Sports-quant tracks upsets across the entire ensemble:

```python
def track_popular_upsets(all_model_data, backtest_teams):
    """Find which upsets are predicted by the most models."""
    upset_counts = {}
    for model_data in all_model_data:
        preds = model_data["y_backtest_pred"]
        for i, row in backtest_teams.iterrows():
            if is_upset_prediction(row, preds[i]):
                key = (row["Team1"], row["Seed1"], row["Team2"], row["Seed2"])
                upset_counts[key] = upset_counts.get(key, 0) + 1
    
    # Sort by frequency — upsets predicted by 80%+ of models are high confidence
    return sorted(upset_counts.items(), key=lambda x: x[1], reverse=True)
```

This is valuable because:
- An upset predicted by 45/50 models is much more credible than one predicted by 3/50
- Shows model consensus vs divergence on controversial picks

---

## Part 4: Survivor Pool Optimizer

Sports-quant includes a survivor pool optimizer with multiple strategies:

### Greedy Strategy
Pick the highest-probability survivor each round.

### Bracket-Aware Strategy  
Avoid picking teams from the same bracket side (so a single upset doesn't eliminate your pool).

### Monte Carlo Optimal Strategy
Simulate thousands of tournament outcomes and find the pick sequence that maximizes expected survival probability.

### Day-Based Mode
Sports-quant's latest feature: 9-slot day-based survivor pools where you need to pick one game from each day of the tournament.

---

## Part 5: Bracket Visualization

Sports-quant has a full SVG bracket rendering pipeline (`_bracket_render_svg.py`, `_bracket_layout.py`, `_bracket_theme.py`). While we have some bracket visualization, their approach is more polished:

- Standard 64-team bracket layout
- Color-coded by prediction confidence
- Correct/incorrect indicators for backtested brackets
- Exportable SVG format

---

## Acceptance Criteria

- [ ] Forward simulation fills bracket round-by-round
- [ ] Monte Carlo mode with pre-computed pairwise probabilities
- [ ] Deterministic mode for single-bracket output
- [ ] Popular upset tracking across ensemble
- [ ] Per-round accuracy reporting (R64, R32, S16, E8, F4, NCG separately)
- [ ] ESPN bracket scoring integrated with simulation
- [ ] Favorites baseline computed alongside model predictions

---

## Dependencies

- Matchup feature computation (needs to work for arbitrary team pairings, not just historical)
- Pre-trained models accessible for inference
- Tournament bracket structure (matchup assignments by region)
