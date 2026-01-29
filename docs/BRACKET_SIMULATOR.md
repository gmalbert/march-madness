# Bracket Simulator

Monte Carlo simulation engine for predicting full tournament outcomes.

## Overview

The Bracket Simulator uses Monte Carlo methods to simulate thousands of tournament brackets, calculating probabilities for each team to reach:
- Final Four
- Championship Game  
- Win the Tournament

## Quick Start

```python
from bracket_simulation import simulate_bracket

# Define tournament predictions with probabilities
predictions = {
    "teams": [
        {"name": "Duke", "seed": 1, "region": "East"},
        {"name": "UConn", "seed": 1, "region": "South"},
        # ... more teams
    ],
    "elite8": [
        {"team1": "Duke", "team2": "Alabama", "team1_prob": 0.65},
        {"team1": "UConn", "team2": "Tennessee", "team1_prob": 0.70},
        # ... more matchups
    ],
    "final4": [
        {"team1": "Duke", "team2": "Kansas", "team1_prob": 0.52},
        {"team1": "UConn", "team2": "Purdue", "team1_prob": 0.58},
    ],
    "championship": [
        {"team1": "Duke", "team2": "UConn", "team1_prob": 0.48},
    ]
}

# Run simulation
results = simulate_bracket(predictions, num_sims=10000)

# View results
for team, probs in results.items():
    print(f"{team}: {probs['winner_pct']:.1%} to win tournament")
```

## Features

### 1. Simple API

**`simulate_bracket(predictions, num_sims)`**
- Runs Monte Carlo simulation of tournament
- Returns probabilities for all teams
- Parameters:
  - `predictions`: Dict with teams and matchup probabilities
  - `num_sims`: Number of simulations (default: 10,000)

**`run_single_simulation(predictions)`**
- Simulates one complete bracket
- Returns winners from each round
- Useful for generating sample brackets

### 2. Advanced Simulation Engine

The module also includes a sophisticated `BracketSimulator` class for complex scenarios:

```python
from bracket_simulation import BracketState, BracketSimulator, create_bracket_from_data

# Create bracket from team data
team_data = {
    "teams": [
        {
            "name": "Duke", 
            "seed": 1, 
            "region": "East",
            "stats": {"net_efficiency": 25.0}
        },
        # ... 64 teams total
    ]
}

bracket_state, simulator = create_bracket_from_data(team_data)

# Run simulation
results = simulator.simulate_bracket(bracket_state, num_simulations=10000)
```

### 3. Integration with Prediction Models

```python
from bracket_simulation import create_predictor_from_models

# Use your ML models to predict game outcomes
predictor = create_predictor_from_models(
    models={'moneyline': my_models},
    efficiency_data=team_efficiency_df
)

# Create simulator with custom predictor
simulator = BracketSimulator(game_predictor=predictor)
```

## Use Cases

### 1. Championship Futures Betting

Identify value in championship odds:

```python
results = simulate_bracket(predictions, num_sims=25000)

for team, probs in sorted(results.items(), 
                          key=lambda x: x[1]['winner_pct'], 
                          reverse=True)[:5]:
    implied_odds = 1 / probs['winner_pct']
    print(f"{team}: {probs['winner_pct']:.1%} (fair odds: +{int((implied_odds-1)*100)})")
```

### 2. Final Four Brackets

Find teams most likely to reach Final Four:

```python
final_four_locks = sorted(
    results.items(),
    key=lambda x: x[1]['final_four_pct'],
    reverse=True
)[:4]

print("Most Likely Final Four:")
for team, probs in final_four_locks:
    print(f"  {team}: {probs['final_four_pct']:.1%}")
```

### 3. Upset Detection

Identify lower seeds with realistic championship chances:

```python
# Find #5 seeds or lower with >2% title chance
upsets = [
    (team, probs, team_data[team]['seed'])
    for team, probs in results.items()
    if team_data[team]['seed'] >= 5 and probs['winner_pct'] > 0.02
]

for team, probs, seed in sorted(upsets, key=lambda x: x[1]['winner_pct'], reverse=True):
    print(f"#{seed} {team}: {probs['winner_pct']:.1%} to win")
```

### 4. Bracket Strategy

Optimize bracket picks by understanding variance:

```python
# High-confidence picks (>80% to advance)
safe_picks = [
    team for team, probs in results.items()
    if probs['final_four_pct'] > 0.80
]

# Differentiation picks (40-60% range)
toss_ups = [
    team for team, probs in results.items()
    if 0.40 < probs['championship_pct'] < 0.60
]
```

## Output Format

`simulate_bracket()` returns a dictionary:

```python
{
    "Duke": {
        "final_four": 7234,           # Raw count
        "championship": 5821,         # Raw count
        "winner": 5234,               # Raw count
        "final_four_pct": 0.7234,     # Probability
        "championship_pct": 0.5821,   # Probability
        "winner_pct": 0.5234          # Probability
    },
    "UConn": {
        ...
    },
    ...
}
```

## Performance

- **10,000 simulations**: ~1-2 seconds (good for quick analysis)
- **50,000 simulations**: ~5-8 seconds (more accurate probabilities)
- **100,000 simulations**: ~10-15 seconds (maximum precision)

For most use cases, 10,000-25,000 simulations provides good balance of speed and accuracy.

## Examples

See demonstration scripts:
- `test_bracket_simulator.py` - Basic usage with simplified bracket
- `demo_full_bracket.py` - Full 16-team bracket with all rounds
- `demo_bracket_betting.py` - Integration with betting strategy

## Validation

Simulations are validated to ensure:
- Total Final Four probability ≈ 4.0 (4 teams make it)
- Total Championship probability ≈ 2.0 (2 teams make it)
- Total Winner probability = 1.0 (exactly 1 winner)

## Advanced Features

### Custom Game Predictors

Define your own win probability function:

```python
def my_predictor(team1, team2):
    # Your custom logic
    eff_diff = team1.stats['net_efficiency'] - team2.stats['net_efficiency']
    return 0.5 + (eff_diff * 0.025)

simulator = BracketSimulator(game_predictor=my_predictor)
```

### Regional Analysis

Track performance by region:

```python
from collections import defaultdict

regional_winners = defaultdict(int)

for _ in range(10000):
    bracket = run_single_simulation(predictions)
    for team in bracket['final_four']:
        regional_winners[team_data[team]['region']] += 1

for region, count in regional_winners.items():
    print(f"{region}: {count/10000:.1%} of Final Four teams")
```

## Implementation Details

- Uses Python's `random` module for Monte Carlo sampling
- Each simulation independently samples win probabilities
- Probabilities are converted to frequencies across all simulations
- No machine learning required (works with any probability source)

## Related Features

- **Upset Predictor**: `upset_prediction.py` for identifying upset candidates
- **Tournament Models**: `tournament_models.py` for ML-based win probabilities
- **Parlay Builder**: `features.py` for multi-game betting strategies

## Roadmap Status

✅ **IMPLEMENTED** - Priority P3 feature from `roadmap-extended-features.md`

All core functionality complete:
- Monte Carlo simulation engine
- Simple API matching roadmap spec
- Advanced bracket state management
- Integration with prediction models
- Comprehensive test coverage
