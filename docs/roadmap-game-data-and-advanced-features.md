# Roadmap: Game-by-Game Data & Advanced Features

> **Priority**: P2 — Unlocks multiple high-value feature classes  
> **Estimated Effort**: High  
> **Source**: sports-quant Kaggle enhancement spec Phase 2 + Phase 5, own analysis  
> **Impact**: Enables Elo, momentum, four factors, conference tournament signals

---

## Problem Statement

Our current data is **season-aggregated** — we have end-of-season KenPom and BartTorvik ratings, but no game-by-game results. This blocks several high-value feature classes:

1. **Elo ratings** — need game-by-game results to compute
2. **Recent form / momentum** — need last 5-10 game results
3. **Four Factors** — need game-level box score data
4. **Conference tournament performance** — need post-conference-tournament results
5. **Rest days** — need game dates to compute days between games

Sports-quant's Kaggle analysis identifies game-by-game data as the key infrastructure unlock for Phase 2 features.

---

## Part 1: Data Sources

### Option A: Kaggle March Machine Learning Mania Dataset (Easiest)
- Free with Kaggle competition signup
- Contains game-by-game results from 2003-present
- Includes: date, team IDs, scores, location, overtime
- Does NOT include box scores (shooting stats, rebounds, etc.)

### Option B: Sports Reference / College Basketball Reference
- Comprehensive box scores
- Requires scraping with proxy rotation (rate-limited)
- Contains four factors, advanced stats, game logs

### Option C: NCAA Stats API
- Official source
- Can be unreliable/slow
- Good for recent seasons

### Recommendation
Start with Kaggle data (covers Elo and momentum). Add Sports Reference later for four factors.

---

## Part 2: Custom Elo Rating System

### Design (following FiveThirtyEight / sports-quant Kaggle spec)

```
Initial rating: 1500 per team
Season regression: 0.85 × prev_elo + 0.15 × league_mean
Home court: ±50 points
K-factor schedule:
  - Early season (days 1-50): K=50 (fast adaptation)
  - Mid season (days 50-100): K=40
  - Late season (days 100+): K=15 (stable)
Cross-conference multiplier: 1.75x
```

### Derived Features (as differences)

| Feature | Description |
|---------|-------------|
| `elo_diff` | Current Elo gap between teams |
| `elo_trend_diff` | Elo change over last 10 games (momentum) |
| `elo_sos_diff` | Average opponent Elo faced (schedule quality) |

### Implementation

```python
class EloSystem:
    def __init__(self, k_early=50, k_mid=40, k_late=15, regression=0.85):
        self.ratings = {}  # {(team, season): current_elo}
    
    def update(self, team_a, team_b, score_a, score_b, day_of_season, is_neutral):
        k = self._get_k(day_of_season)
        expected_a = 1 / (1 + 10 ** ((self.ratings[team_b] - self.ratings[team_a]) / 400))
        actual_a = 1 if score_a > score_b else 0
        self.ratings[team_a] += k * (actual_a - expected_a)
        self.ratings[team_b] += k * ((1 - actual_a) - (1 - expected_a))
    
    def regress_to_mean(self, season):
        """Apply between-season regression."""
        league_mean = np.mean(list(self.ratings.values()))
        for team in self.ratings:
            self.ratings[team] = self.regression * self.ratings[team] + (1-self.regression) * league_mean
```

---

## Part 3: Dean Oliver's Four Factors

The four factors explain 98% of offensive efficiency variance:

| Factor | Formula | Weight (approx) |
|--------|---------|-----------------|
| **eFG%** | (FGM + 0.5 × 3PM) / FGA | 40% |
| **TOV%** | Turnovers / possessions | 25% |
| **OREB%** | OReb / (OReb + Opp DReb) | 20% |
| **FT Rate** | FTA / FGA | 15% |

### Features (all as differences)
```
efg_pct_diff    — Effective Field Goal % gap
tov_pct_diff    — Turnover Rate gap
oreb_pct_diff   — Offensive Rebound % gap
ft_rate_diff    — Free Throw Rate gap
```

### Data Requirement
Needs season-level box score aggregates. Available from:
- Sports Reference (scraping)
- Kaggle detailed results (partial)
- BartTorvik (may already include)

---

## Part 4: Recent Form / Momentum Features

Captures trajectory in the weeks before the tournament:

| Feature | Description |
|---------|-------------|
| `last_5_margin_diff` | Average scoring margin over last 5 games |
| `last_10_win_pct_diff` | Win rate over last 10 games |
| `conf_tourney_result` | Conference tournament result (1.0=won, 0.5=final, 0.25=semi, 0=early loss) |
| `days_since_last_game` | Rest advantage |

---

## Part 5: Tournament-Specific Features

| Feature | Description |
|---------|-------------|
| `coach_tourney_appearances_diff` | Coach tournament experience |
| `program_seed_history` | How the program historically performs at this seed |
| `seed_matchup_winrate` | Empirical P(higher seed wins) for this specific pairing |
| `travel_distance_diff` | Distance to game site (FiveThirtyEight adjustment) |

---

## Part 6: Composite Power Ratings

FiveThirtyEight's insight: averaging 4-6 independent rating systems outperforms any single one.

### Rating Systems to Ingest
- KenPom (already have)
- BartTorvik (already have)
- Sagarin (via Massey Composite)
- NET Rankings (already have partial)
- ESPN BPI
- Massey Ratings

### Implementation
```python
composite_rating = np.mean([kenpom_rating, barttorvik_rating, sagarin_rating, ...])
composite_rating_diff = team1_composite - team2_composite
```

Missing ratings handled natively by tree-based models (NaN support).

---

## Implementation Priority

```
Phase A (data infrastructure):
  1. Ingest Kaggle game-by-game data
  2. Build Elo system
  3. Compute recent form features

Phase B (box score features):
  4. Add four factors from Sports Reference or BartTorvik
  5. Conference tournament results

Phase C (external ratings):
  6. Massey Composite ingestion
  7. Composite power rating feature

Phase D (contextual features):
  8. Coach experience
  9. Travel distance
  10. Program historical performance
```

---

## Acceptance Criteria

- [ ] Game-by-game results ingested for 2010-present (minimum)
- [ ] Elo system computing correct ratings verified against FiveThirtyEight
- [ ] At least 3 new feature classes integrated (Elo, momentum, four factors)
- [ ] Backtest confirms additive value of new features
- [ ] Composite power rating from 3+ independent sources

---

## Dependencies

- Difference features and temporal CV (should be in place first)
- External data access (Kaggle signup, possibly proxy rotation for scraping)
