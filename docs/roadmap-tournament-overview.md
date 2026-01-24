# Roadmap: Tournament Prediction Overview

*Master plan for predicting every winner of every round in March Madness.*

## Project Goals

1. **Predict every game** in the NCAA Tournament (63 games for men's, 63 for women's)
2. **Calculate probabilities** for teams reaching each round (Sweet 16, Elite 8, Final Four, Championship)
3. **Visualize predictions** in an interactive bracket format
4. **Identify Cinderella teams** and potential upsets
5. **Track prediction accuracy** as tournament progresses

## Tournament Structure

```
Round of 64 (First Round)     → 32 games
Round of 32 (Second Round)    → 16 games  
Sweet 16                      → 8 games
Elite 8                       → 4 games
Final Four                    → 2 games
Championship                  → 1 game
─────────────────────────────────────────
Total                         → 63 games
```

## Bracket Regions

```python
REGIONS = {
    'East': {'location': 'TBD', 'seeds': list(range(1, 17))},
    'West': {'location': 'TBD', 'seeds': list(range(1, 17))},
    'South': {'location': 'TBD', 'seeds': list(range(1, 17))},
    'Midwest': {'location': 'TBD', 'seeds': list(range(1, 17))}
}

# First round matchups (by seed)
FIRST_ROUND_MATCHUPS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15)
]
```

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MARCH MADNESS PREDICTOR                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ Data Layer   │───▶│ Model Layer  │───▶│ Simulation   │       │
│  │              │    │              │    │ Engine       │       │
│  │ • Team Stats │    │ • Win Prob   │    │              │       │
│  │ • Efficiency │    │ • Spread     │    │ • Monte Carlo│       │
│  │ • Rankings   │    │ • Upset Prob │    │ • 10K+ sims  │       │
│  │ • Historical │    │              │    │              │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                   │                │
│         ▼                   ▼                   ▼                │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                  Bracket Generator                    │       │
│  │  • Round-by-round predictions                        │       │
│  │  • Probability distributions                          │       │
│  │  • Upset alerts                                       │       │
│  └──────────────────────────────────────────────────────┘       │
│                            │                                     │
│                            ▼                                     │
│  ┌──────────────────────────────────────────────────────┐       │
│  │                  Visualization Layer                  │       │
│  │  • Interactive bracket                                │       │
│  │  • Probability heatmaps                              │       │
│  │  • Path to championship                              │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Roadmap Documents

| Document | Purpose |
|----------|---------|
| `roadmap-tournament-data.md` | Data sources and collection for tournament |
| `roadmap-tournament-models.md` | Prediction models for tournament games |
| `roadmap-bracket-simulation.md` | Monte Carlo simulation for full bracket |
| `roadmap-bracket-visualization.md` | Bracket UI and visualization |
| `roadmap-upset-detection.md` | Identifying potential upsets |
| `roadmap-implementation.md` | Step-by-step implementation plan |

## Key Metrics

```python
# Prediction accuracy metrics
METRICS = {
    'game_accuracy': 'Percentage of games predicted correctly',
    'upset_detection': 'Percentage of upsets correctly predicted',
    'final_four_accuracy': 'Final Four teams correctly predicted',
    'champion_accuracy': 'Championship winner predicted correctly',
    'bracket_score': 'ESPN-style scoring (10-20-40-80-160-320)',
    'log_loss': 'Probabilistic accuracy of predictions'
}

# ESPN Bracket Scoring
ROUND_POINTS = {
    'Round of 64': 10,
    'Round of 32': 20,
    'Sweet 16': 40,
    'Elite 8': 80,
    'Final Four': 160,
    'Championship': 320
}
# Perfect bracket = 1920 points
```

## Timeline

```
Pre-Tournament (Selection Sunday - Thursday):
├── Collect final season stats
├── Get official bracket/seeding
├── Generate initial predictions
└── Run Monte Carlo simulations

Tournament Week 1 (Thursday - Sunday):
├── Track Round of 64 results
├── Update predictions with actual results
└── Recalculate remaining probabilities

Tournament Week 2 (Thursday - Sunday):
├── Track Sweet 16 and Elite 8
├── Refine Final Four predictions
└── Update bracket visualization

Final Weekend:
├── Final Four predictions
├── Championship prediction
└── Final accuracy assessment
```

## Next Steps

1. See `roadmap-tournament-data.md` for data collection
2. See `roadmap-tournament-models.md` for model architecture
3. See `roadmap-bracket-simulation.md` for simulation engine
4. See `roadmap-bracket-visualization.md` for UI design
