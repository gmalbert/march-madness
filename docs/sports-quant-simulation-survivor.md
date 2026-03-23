# Sports-Quant Bracket Simulation & Survivor Pool

> Source: [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) — March Madness module  
> Purpose: Document forward simulation engine and survivor pool optimizer from sports-quant for integration into our project.

---

## Executive Summary

| Feature | Our Current State | sports-quant Approach | Impact |
|---------|-------------------|----------------------|--------|
| Simulation | 1000 Monte Carlo sims, basic matchups | Forward sim with FeatureLookup for arbitrary matchups | Medium |
| Debiased game prediction | None | Column-swap averaging per game | High |
| Cascading error model | Partially (our sim does forward) | Explicit forward prop with accuracy tracking by round | Medium |
| Survivor pool | Not implemented | Greedy + optimal + Monte Carlo strategies | New capability |
| Live prediction | Not implemented | Combines known results + MC forward sim | New capability |

---

## 1. Forward Simulation Architecture

### The Limitation of Our Current Backtest
Our current system evaluates model accuracy on actual games that occurred. For later rounds (R32+), the model gets "free" knowledge of which teams actually advanced. The model never faces consequences of wrong earlier predictions.

### Forward Simulation Fix
Forward simulation starts with known R64 matchups and **advances winners to form next-round matchups**. If the model gets R64 wrong, it creates different R32 matchups — and the error cascades.

```
R64: Model predicts all 32 games
     ↓ advance winners
R32: Matchups formed from R64 winners (may differ from reality)
     ↓ advance winners
S16: Matchups formed from R32 winners
     ↓ ... through championship
```

### Core Prediction Function

```python
# bracket_forward_sim.py — Forward simulation engine

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TeamStats:
    """Pre-tournament stats for a single team."""
    team: str
    year: int
    seed: int
    features: dict  # column_name -> value


@dataclass(frozen=True)
class SimulationResult:
    """Result of a single deterministic bracket simulation."""
    year: int
    games: list[dict]           # All 63 predicted games
    accuracy_by_round: dict     # round -> (correct, total)
    overall_accuracy: tuple     # (correct, total)


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated results from N Monte Carlo simulations."""
    year: int
    n_simulations: int
    team_round_rates: dict      # team -> {round -> advance fraction}
    mean_accuracy_by_round: dict  # round -> mean accuracy
    champion_distribution: dict   # team -> championship fraction


def predict_game(
    team1: TeamStats,
    team2: TeamStats,
    models: list,
    feature_lookup,  # FeatureLookup instance
) -> float:
    """Predict Team1 win probability with debiasing.
    
    1. Build feature vector for (team1, team2)
    2. Get probability from each model
    3. Swap team1/team2 features (negate for diff features)
    4. Get swapped probability from each model
    5. Average original and (1 - swapped)
    
    Returns:
        Debiased probability that team1 wins.
    """
    # Build feature vectors
    X_original = feature_lookup.build_difference_features(
        team1.team, team2.team, team1.year,
        seed1=team1.seed, seed2=team2.seed,
    )
    
    if X_original is None:
        # Fallback: use seed-based prior
        prior = _seed_prior(team1.seed, team2.seed)
        return prior
    
    # Predictions on original features
    original_probs = [m.predict_proba(X_original)[:, 1][0] for m in models]
    avg_original = np.mean(original_probs)
    
    # Predictions on swapped features (negate for diff features)
    X_swapped = X_original * -1
    swapped_probs = [1 - m.predict_proba(X_swapped)[:, 1][0] for m in models]
    avg_swapped = np.mean(swapped_probs)
    
    # Average for debiased probability
    return (avg_original + avg_swapped) / 2


def _seed_prior(seed1: int, seed2: int) -> float:
    """Fallback seed-based win probability."""
    PRIORS = {
        (1, 16): 0.99, (2, 15): 0.94, (3, 14): 0.85, (4, 13): 0.79,
        (5, 12): 0.65, (6, 11): 0.62, (7, 10): 0.61, (8, 9): 0.52,
    }
    key = (min(seed1, seed2), max(seed1, seed2))
    prior = PRIORS.get(key, 0.5)
    return prior if seed1 <= seed2 else 1 - prior
```

---

## 2. Deterministic Simulation

Always picks the team with probability > 0.5:

```python
# Canonical seed pairings for bracket structure
CANONICAL_SEED_PAIRS = [
    (1, 16), (8, 9), (5, 12), (4, 13),
    (6, 11), (3, 14), (7, 10), (2, 15),
]
# Adjacent pairs feed into next round: slots 0+1 -> R32 game 0, etc.


def simulate_bracket_deterministic(
    year: int,
    models: list,
    feature_lookup,
    r64_matchups: list[tuple[TeamStats, TeamStats]],
    actual_results: Optional[dict] = None,
) -> SimulationResult:
    """Simulate a full bracket by always picking higher-probability team.
    
    Args:
        year: Tournament year.
        models: Trained prediction models.
        feature_lookup: FeatureLookup for building feature vectors.
        r64_matchups: 32 R64 matchups in canonical bracket order.
        actual_results: Optional dict of round -> list of actual winners.
        
    Returns:
        SimulationResult with 63 predicted games and accuracy.
    """
    all_games = []
    current_matchups = r64_matchups
    
    round_names = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    round_game_counts = [32, 16, 8, 4, 2, 1]
    
    accuracy_by_round = {}
    total_correct = 0
    total_games = 0
    
    for round_idx, (round_name, n_games) in enumerate(
        zip(round_names, round_game_counts)
    ):
        winners = []
        correct_this_round = 0
        
        for game_idx, (team1, team2) in enumerate(current_matchups):
            prob = predict_game(team1, team2, models, feature_lookup)
            winner = team1 if prob > 0.5 else team2
            
            game = {
                "round": round_name,
                "game_index": game_idx,
                "team1": team1.team,
                "seed1": team1.seed,
                "team2": team2.team,
                "seed2": team2.seed,
                "predicted_winner": winner.team,
                "win_probability": prob if winner == team1 else 1 - prob,
            }
            
            # Check accuracy against actual results
            if actual_results and round_name in actual_results:
                actual_winner = actual_results[round_name][game_idx]
                game["actual_winner"] = actual_winner
                game["is_correct"] = (winner.team == actual_winner)
                if game["is_correct"]:
                    correct_this_round += 1
            
            all_games.append(game)
            winners.append(winner)
        
        accuracy_by_round[round_name] = (correct_this_round, n_games)
        total_correct += correct_this_round
        total_games += n_games
        
        # Form next round's matchups from adjacent winners
        current_matchups = [
            (winners[i], winners[i + 1])
            for i in range(0, len(winners), 2)
        ]
    
    return SimulationResult(
        year=year,
        games=all_games,
        accuracy_by_round=accuracy_by_round,
        overall_accuracy=(total_correct, total_games),
    )
```

---

## 3. Monte Carlo Simulation

Same as deterministic, but samples outcomes from probabilities:

```python
def simulate_bracket_monte_carlo(
    year: int,
    models: list,
    feature_lookup,
    r64_matchups: list[tuple[TeamStats, TeamStats]],
    actual_results: Optional[dict] = None,
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> MonteCarloResult:
    """Run N bracket simulations, sampling outcomes from model probabilities.
    
    If model gives Team A a 72% chance, Team A advances in ~72% of sims.
    
    Returns:
        MonteCarloResult with advance rates, accuracy distributions,
        and championship probabilities.
    """
    rng = np.random.RandomState(rng_seed)
    
    # Track results across simulations
    team_round_counts = {}  # team -> {round -> count}
    champion_counts = {}
    accuracy_by_round_lists = {r: [] for r in ["R64", "R32", "S16", "E8", "F4", "NCG"]}
    
    # Pre-compute all game probabilities (reuse across sims)
    # Only R64 probs are fixed; later rounds depend on who advances
    # So we cache R64 probs and recompute later rounds per sim
    
    for sim in range(n_simulations):
        current_matchups = list(r64_matchups)
        round_names = ["R64", "R32", "S16", "E8", "F4", "NCG"]
        
        for round_name in round_names:
            winners = []
            correct = 0
            
            for game_idx, (team1, team2) in enumerate(current_matchups):
                prob = predict_game(team1, team2, models, feature_lookup)
                
                # Sample outcome from probability
                if rng.random() < prob:
                    winner = team1
                else:
                    winner = team2
                
                # Track team advancement
                key = winner.team
                if key not in team_round_counts:
                    team_round_counts[key] = {}
                team_round_counts[key][round_name] = (
                    team_round_counts[key].get(round_name, 0) + 1
                )
                
                # Check accuracy
                if actual_results and round_name in actual_results:
                    if winner.team == actual_results[round_name][game_idx]:
                        correct += 1
                
                winners.append(winner)
            
            accuracy_by_round_lists[round_name].append(correct)
            
            # Form next round matchups
            current_matchups = [
                (winners[i], winners[i + 1])
                for i in range(0, len(winners), 2)
            ]
        
        # Track champion
        champion = winners[0].team
        champion_counts[champion] = champion_counts.get(champion, 0) + 1
    
    # Normalize counts to fractions
    team_round_rates = {}
    for team, rounds in team_round_counts.items():
        team_round_rates[team] = {
            r: count / n_simulations for r, count in rounds.items()
        }
    
    champion_dist = {
        t: c / n_simulations for t, c in champion_counts.items()
    }
    
    mean_accuracy = {
        r: np.mean(counts) for r, counts in accuracy_by_round_lists.items()
    }
    
    return MonteCarloResult(
        year=year,
        n_simulations=n_simulations,
        team_round_rates=team_round_rates,
        mean_accuracy_by_round=mean_accuracy,
        champion_distribution=champion_dist,
    )
```

---

## 4. Survivor Pool Optimizer

### Overview
Pick **one team per round** to win. Once used, that team is burned. If your pick loses, you're eliminated. Optimal strategy requires saving strong favorites for rounds where they're needed most.

### Data Structures

```python
from dataclasses import dataclass
from typing import FrozenSet


@dataclass(frozen=True)
class SurvivorPick:
    """A single survivor pool pick."""
    round_name: str
    team: str
    seed: int
    opponent: str
    opponent_seed: int
    win_probability: float
    actual_winner: str
    survived: bool


@dataclass(frozen=True)
class SurvivorResult:
    """Complete survivor pool result for one tournament."""
    year: int
    strategy: str      # "greedy" or "optimal"
    picks: tuple[SurvivorPick, ...]  # 6 picks
    survived_all: bool
    rounds_survived: int
    survival_probability: float  # Product of all pick win probs
```

### Greedy Strategy

```python
def run_survivor_greedy(
    year: int,
    game_probabilities: pd.DataFrame,
) -> SurvivorResult:
    """Pick highest win-prob unused team each round.
    
    Simple but suboptimal: might burn a strong team early
    when it's needed more in a later round.
    
    Args:
        game_probabilities: DataFrame with columns:
            Team1, Seed1, Team2, Seed2, CURRENT_ROUND,
            Win_Prob (Team1 win probability), Team1_Win (actual)
    """
    round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    used_teams = set()
    picks = []
    
    for round_name in round_order:
        round_games = game_probabilities[
            game_probabilities["CURRENT_ROUND"] == round_name
        ]
        
        # Build candidate list: both teams from each game, with win probs
        candidates = []
        for _, game in round_games.iterrows():
            t1_prob = game["Win_Prob"]
            t2_prob = 1 - t1_prob
            
            if game["Team1"] not in used_teams:
                candidates.append({
                    "team": game["Team1"], "seed": game["Seed1"],
                    "opponent": game["Team2"], "opp_seed": game["Seed2"],
                    "prob": t1_prob,
                    "actual_winner": game["Team1"] if game["Team1_Win"] == 1 else game["Team2"],
                })
            if game["Team2"] not in used_teams:
                candidates.append({
                    "team": game["Team2"], "seed": game["Seed2"],
                    "opponent": game["Team1"], "opp_seed": game["Seed1"],
                    "prob": t2_prob,
                    "actual_winner": game["Team1"] if game["Team1_Win"] == 1 else game["Team2"],
                })
        
        if not candidates:
            break
        
        # Pick the highest probability candidate
        best = max(candidates, key=lambda c: c["prob"])
        survived = best["team"] == best["actual_winner"]
        
        picks.append(SurvivorPick(
            round_name=round_name,
            team=best["team"], seed=best["seed"],
            opponent=best["opponent"], opponent_seed=best["opp_seed"],
            win_probability=best["prob"],
            actual_winner=best["actual_winner"],
            survived=survived,
        ))
        
        used_teams.add(best["team"])
        
        if not survived:
            break  # Eliminated
    
    survival_prob = 1.0
    for p in picks:
        survival_prob *= p.win_probability
    
    return SurvivorResult(
        year=year,
        strategy="greedy",
        picks=tuple(picks),
        survived_all=all(p.survived for p in picks),
        rounds_survived=sum(1 for p in picks if p.survived),
        survival_probability=survival_prob,
    )
```

### Optimal Strategy (Branch-and-Bound)

```python
def run_survivor_optimal(
    year: int,
    game_probabilities: pd.DataFrame,
) -> SurvivorResult:
    """Find the pick sequence that maximizes P(survive all 6 rounds).
    
    Uses branch-and-bound pruning: abandon any partial sequence
    whose probability product is already below the current best.
    
    Search space: ~64 × 32 × 16 × 8 × 4 × 2 candidates per level,
    heavily prunable. Runs in milliseconds.
    """
    round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    
    # Pre-build candidates per round
    round_candidates = {}
    for round_name in round_order:
        round_games = game_probabilities[
            game_probabilities["CURRENT_ROUND"] == round_name
        ]
        candidates = []
        for _, game in round_games.iterrows():
            t1_prob = game["Win_Prob"]
            actual_winner = game["Team1"] if game["Team1_Win"] == 1 else game["Team2"]
            
            candidates.append({
                "team": game["Team1"], "seed": game["Seed1"],
                "opponent": game["Team2"], "opp_seed": game["Seed2"],
                "prob": t1_prob, "actual_winner": actual_winner,
            })
            candidates.append({
                "team": game["Team2"], "seed": game["Seed2"],
                "opponent": game["Team1"], "opp_seed": game["Seed1"],
                "prob": 1 - t1_prob, "actual_winner": actual_winner,
            })
        round_candidates[round_name] = candidates
    
    best_prob = [0.0]
    best_sequence = [None]
    
    def search(round_idx: int, used: frozenset, current_prob: float, path: list):
        if round_idx == len(round_order):
            if current_prob > best_prob[0]:
                best_prob[0] = current_prob
                best_sequence[0] = list(path)
            return
        
        round_name = round_order[round_idx]
        for cand in round_candidates[round_name]:
            if cand["team"] in used:
                continue
            
            new_prob = current_prob * cand["prob"]
            
            # Prune: even if all remaining rounds have prob=1.0,
            # this path can't beat current best
            if new_prob <= best_prob[0]:
                continue
            
            path.append(cand)
            search(round_idx + 1, used | {cand["team"]}, new_prob, path)
            path.pop()
    
    search(0, frozenset(), 1.0, [])
    
    if best_sequence[0] is None:
        raise ValueError("No valid survivor sequence found")
    
    picks = []
    for cand, round_name in zip(best_sequence[0], round_order):
        survived = cand["team"] == cand["actual_winner"]
        picks.append(SurvivorPick(
            round_name=round_name,
            team=cand["team"], seed=cand["seed"],
            opponent=cand["opponent"], opponent_seed=cand["opp_seed"],
            win_probability=cand["prob"],
            actual_winner=cand["actual_winner"],
            survived=survived,
        ))
    
    return SurvivorResult(
        year=year,
        strategy="optimal",
        picks=tuple(picks),
        survived_all=all(p.survived for p in picks),
        rounds_survived=sum(1 for p in picks if p.survived),
        survival_probability=best_prob[0],
    )
```

### Live Prediction Mode

Combines known results with Monte Carlo forward simulation:

```python
@dataclass(frozen=True)
class LiveSurvivorState:
    """State of a survivor pool in progress."""
    year: int
    completed_rounds: tuple[str, ...]
    picks_made: tuple[SurvivorPick, ...]
    teams_used: frozenset
    still_alive: bool


def run_survivor_live(
    state: LiveSurvivorState,
    models: list,
    feature_lookup,
    known_matchups: pd.DataFrame,
    n_simulations: int = 1000,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Recommend a survivor pick for the next round of a live tournament.
    
    Algorithm:
        1. Identify the next round to pick for
        2. Get actual matchups for that round
        3. For each candidate team (not yet used):
           a. Assume we pick this team
           b. Run N MC forward sims of ALL remaining rounds
           c. Record: fraction of sims where we survive ALL remaining rounds
        4. Return: team -> P(survive rest of tournament | pick this team)
    
    Returns:
        Dict of team_name -> probability of surviving all remaining rounds.
    """
    round_order = ["R64", "R32", "S16", "E8", "F4", "NCG"]
    next_round_idx = len(state.completed_rounds)
    
    if next_round_idx >= len(round_order):
        return {}  # Tournament is over
    
    rng = np.random.RandomState(rng_seed)
    next_round = round_order[next_round_idx]
    
    # Get matchups for the next round
    next_matchups = known_matchups[
        known_matchups["CURRENT_ROUND"] == next_round
    ]
    
    # Build candidate list
    candidates = set()
    for _, game in next_matchups.iterrows():
        if game["Team1"] not in state.teams_used:
            candidates.add(game["Team1"])
        if game["Team2"] not in state.teams_used:
            candidates.add(game["Team2"])
    
    # For each candidate, simulate remaining tournament N times
    results = {}
    for candidate in candidates:
        survived_count = 0
        
        for _ in range(n_simulations):
            # Check if our pick survives this round (sample from prob)
            game = next_matchups[
                (next_matchups["Team1"] == candidate) |
                (next_matchups["Team2"] == candidate)
            ].iloc[0]
            
            if game["Team1"] == candidate:
                prob = game["Win_Prob"]
            else:
                prob = 1 - game["Win_Prob"]
            
            if rng.random() >= prob:
                continue  # Pick lost this round
            
            # Forward simulate remaining rounds with greedy strategy
            # (simplified — full implementation would use predict_game)
            survived_remaining = True
            used = state.teams_used | {candidate}
            
            for future_round_idx in range(next_round_idx + 1, len(round_order)):
                future_round = round_order[future_round_idx]
                future_games = known_matchups[
                    known_matchups["CURRENT_ROUND"] == future_round
                ]
                
                # Find best available pick
                best_prob = 0
                for _, fg in future_games.iterrows():
                    for t, p in [(fg["Team1"], fg["Win_Prob"]), 
                                 (fg["Team2"], 1 - fg["Win_Prob"])]:
                        if t not in used and p > best_prob:
                            best_prob = p
                
                # Sample whether best pick survives
                if rng.random() >= best_prob:
                    survived_remaining = False
                    break
            
            if survived_remaining:
                survived_count += 1
        
        results[candidate] = survived_count / n_simulations
    
    return results
```

---

## 5. Evaluation & Output Formats

### Deterministic Bracket CSV
```
Round, GameIndex, Region, Team1, Seed1, Team2, Seed2, PredictedWinner, WinProbability, ActualWinner, IsCorrect
```

### Monte Carlo Summary
```
Team, Seed, R64, R32, S16, E8, F4, NCG, Champion
```
(Values = fraction of simulations where team reached/won that round)

### Survivor Picks CSV
```
Round, Team, Seed, Opponent, OpponentSeed, WinProbability, ActualWinner, Survived
```

---

## 6. Integration with Our Existing System

Our `bracket_simulation.py` already has Monte Carlo simulation with 10,000 runs. The sports-quant approach adds:

1. **FeatureLookup for arbitrary matchups** — our sim currently needs team objects with stats pre-loaded; the FeatureLookup pattern is more flexible
2. **Debiased game-level predictions** — column-swap averaging for each game prediction
3. **Survivor pool optimizer** — entirely new capability
4. **Live prediction mode** — combines known results with forward simulation
5. **Per-round accuracy tracking** — granular evaluation of where the model struggles

### What We Already Have
- Monte Carlo simulation engine (10K sims)
- Region-based bracket structure
- Team/seed matching
- Probability-based game prediction

### What We're Missing
- Debiased per-game predictions in the sim
- Survivor pool (greedy + optimal + live)
- FeatureLookup for hypothetical matchups
- Formal evaluation framework (per-round accuracy, cascading error measurement)

---

## Priority Implementation Order

1. **Debiased game predictions** — directly improves existing simulation
2. **FeatureLookup integration** — enables arbitrary matchup prediction
3. **Per-round accuracy evaluation** — understand where the model fails
4. **Survivor pool (greedy)** — new user-facing feature
5. **Survivor pool (optimal)** — upper bound analysis
6. **Live prediction mode** — real-time tournament support
