# Roadmap: Bracket Simulation

*Monte Carlo simulation for predicting full tournament outcomes.*

## Why Simulation?

Single-game predictions don't capture the **compounding uncertainty** of a tournament. A team's path to the Final Four depends on:
- Their own performance
- Who they might face (other games' outcomes)
- Variance and upset potential

Monte Carlo simulation runs thousands of brackets to calculate **probability distributions** for each team reaching each round.

## Simulation Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    BRACKET SIMULATION ENGINE                    │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐       │
│  │   Bracket   │────▶│   Game      │────▶│  Outcome    │       │
│  │   State     │     │   Predictor │     │  Sampler    │       │
│  └─────────────┘     └─────────────┘     └─────────────┘       │
│        │                    │                   │               │
│        │              Uses model            Samples from        │
│        │              probability           probability         │
│        ▼                                        │               │
│  ┌─────────────────────────────────────────────▼───────────┐   │
│  │                    Round Simulator                       │   │
│  │  For each round:                                        │   │
│  │    1. Get matchups from current bracket state           │   │
│  │    2. Predict each game                                 │   │
│  │    3. Sample winner based on probability                │   │
│  │    4. Advance winners to next round                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│        │                                                        │
│        ▼                                                        │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Results Aggregator                     │   │
│  │  After N simulations:                                    │   │
│  │    - Count times each team reached each round            │   │
│  │    - Calculate probability distributions                  │   │
│  │    - Identify most likely bracket                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

## Core Simulation Code

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from copy import deepcopy
import random

@dataclass
class Team:
    """Tournament team."""
    id: str
    name: str
    seed: int
    region: str
    stats: Dict = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id

@dataclass
class BracketState:
    """Current state of the tournament bracket."""
    teams: Dict[str, Team]  # All 64 teams
    regions: Dict[str, List[Team]]  # Teams by region
    rounds: Dict[int, List[tuple]]  # Matchups by round
    results: Dict[int, List[Team]]  # Winners by round
    
    def get_remaining_teams(self, round_num: int) -> List[Team]:
        """Get teams still alive entering a round."""
        if round_num == 1:
            return list(self.teams.values())
        return self.results.get(round_num - 1, [])
    
    def get_matchups(self, round_num: int, region: str = None) -> List[tuple]:
        """Get matchups for a specific round."""
        if round_num <= 4:  # Regional rounds
            # Matchups within regions
            matchups = []
            for r, teams in self.regions.items():
                if region and r != region:
                    continue
                remaining = [t for t in teams if t in self.get_remaining_teams(round_num)]
                # Pair teams based on bracket position
                matchups.extend(self._pair_for_round(remaining, round_num))
            return matchups
        else:  # Final Four and Championship
            remaining = self.get_remaining_teams(round_num)
            return self._pair_for_round(remaining, round_num)
    
    def _pair_for_round(self, teams: List[Team], round_num: int) -> List[tuple]:
        """Pair teams for matchups based on tournament bracket rules."""
        # Sort by seed for proper bracket pairing
        sorted_teams = sorted(teams, key=lambda t: t.seed)
        matchups = []
        
        # Standard bracket pairing: best vs worst remaining
        n = len(sorted_teams)
        for i in range(n // 2):
            matchups.append((sorted_teams[i], sorted_teams[n - 1 - i]))
        
        return matchups


class BracketSimulator:
    """Monte Carlo bracket simulator."""
    
    def __init__(self, predictor, seed: int = None):
        """
        Args:
            predictor: Model that can predict game outcomes
            seed: Random seed for reproducibility
        """
        self.predictor = predictor
        self.rng = np.random.default_rng(seed)
    
    def simulate_game(self, team1: Team, team2: Team, 
                      round_num: int) -> Team:
        """Simulate a single game outcome."""
        # Get win probability from predictor
        game_context = {'round_number': round_num}
        prediction = self.predictor.predict_game(
            team1.stats, team2.stats, game_context
        )
        
        prob_team1_wins = prediction['team1_win_prob']
        
        # Sample outcome based on probability
        if self.rng.random() < prob_team1_wins:
            return team1
        return team2
    
    def simulate_round(self, bracket: BracketState, 
                       round_num: int) -> List[Team]:
        """Simulate all games in a round."""
        matchups = bracket.get_matchups(round_num)
        winners = []
        
        for team1, team2 in matchups:
            winner = self.simulate_game(team1, team2, round_num)
            winners.append(winner)
        
        return winners
    
    def simulate_tournament(self, initial_bracket: BracketState) -> Dict:
        """Simulate complete tournament once."""
        bracket = deepcopy(initial_bracket)
        
        round_names = {
            1: 'Round of 64',
            2: 'Round of 32',
            3: 'Sweet 16',
            4: 'Elite 8',
            5: 'Final Four',
            6: 'Championship'
        }
        
        results = {
            'rounds': {},
            'champion': None
        }
        
        for round_num in range(1, 7):
            winners = self.simulate_round(bracket, round_num)
            bracket.results[round_num] = winners
            results['rounds'][round_names[round_num]] = [
                {'team': w.name, 'seed': w.seed, 'region': w.region}
                for w in winners
            ]
            
            # Update regions for regional rounds
            if round_num <= 4:
                for region in bracket.regions:
                    bracket.regions[region] = [
                        t for t in bracket.regions[region] if t in winners
                    ]
        
        # Champion is the single winner of round 6
        results['champion'] = winners[0].name if winners else None
        results['final_four'] = [
            t['team'] for t in results['rounds'].get('Elite 8', [])
        ]
        
        return results
    
    def run_simulations(self, initial_bracket: BracketState, 
                        n_simulations: int = 10000) -> Dict:
        """Run multiple tournament simulations."""
        
        # Track results
        team_round_counts = {
            team_id: {round_num: 0 for round_num in range(1, 7)}
            for team_id in initial_bracket.teams.keys()
        }
        championship_counts = {}
        final_four_counts = {}
        
        for _ in range(n_simulations):
            result = self.simulate_tournament(initial_bracket)
            
            # Count championship wins
            champion = result['champion']
            championship_counts[champion] = championship_counts.get(champion, 0) + 1
            
            # Count Final Four appearances
            for team in result.get('final_four', []):
                final_four_counts[team] = final_four_counts.get(team, 0) + 1
            
            # Count round advancement
            for round_name, winners in result['rounds'].items():
                round_num = {
                    'Round of 64': 1, 'Round of 32': 2,
                    'Sweet 16': 3, 'Elite 8': 4,
                    'Final Four': 5, 'Championship': 6
                }[round_name]
                
                for winner in winners:
                    team_id = winner['team']
                    if team_id in team_round_counts:
                        team_round_counts[team_id][round_num] += 1
        
        # Convert counts to probabilities
        team_probabilities = {}
        for team_id, counts in team_round_counts.items():
            team = initial_bracket.teams[team_id]
            team_probabilities[team_id] = {
                'name': team.name,
                'seed': team.seed,
                'region': team.region,
                'round_32_prob': counts[1] / n_simulations,
                'sweet_16_prob': counts[2] / n_simulations,
                'elite_8_prob': counts[3] / n_simulations,
                'final_four_prob': counts[4] / n_simulations,
                'finals_prob': counts[5] / n_simulations,
                'champion_prob': counts[6] / n_simulations
            }
        
        return {
            'n_simulations': n_simulations,
            'team_probabilities': team_probabilities,
            'championship_distribution': {
                k: v / n_simulations for k, v in championship_counts.items()
            },
            'final_four_distribution': {
                k: v / n_simulations for k, v in final_four_counts.items()
            },
            'most_likely_champion': max(championship_counts, key=championship_counts.get),
            'most_likely_final_four': sorted(
                final_four_counts.keys(), 
                key=lambda x: final_four_counts[x], 
                reverse=True
            )[:4]
        }
```

## Most Likely Bracket

```python
def generate_most_likely_bracket(predictor, initial_bracket: BracketState) -> Dict:
    """
    Generate single bracket choosing most likely winner for each game.
    This is deterministic - always picks the favorite.
    """
    bracket = deepcopy(initial_bracket)
    
    round_names = ['Round of 64', 'Round of 32', 'Sweet 16', 
                   'Elite 8', 'Final Four', 'Championship']
    
    results = {'rounds': {}, 'predictions': []}
    
    for round_num, round_name in enumerate(round_names, 1):
        matchups = bracket.get_matchups(round_num)
        winners = []
        
        for team1, team2 in matchups:
            game_context = {'round_number': round_num}
            prediction = predictor.predict_game(
                team1.stats, team2.stats, game_context
            )
            
            # Pick favorite (highest probability)
            if prediction['team1_win_prob'] > 0.5:
                winner = team1
                loser = team2
                win_prob = prediction['team1_win_prob']
            else:
                winner = team2
                loser = team1
                win_prob = prediction['team2_win_prob']
            
            winners.append(winner)
            results['predictions'].append({
                'round': round_name,
                'winner': winner.name,
                'winner_seed': winner.seed,
                'loser': loser.name,
                'loser_seed': loser.seed,
                'win_probability': win_prob,
                'is_upset': winner.seed > loser.seed
            })
        
        bracket.results[round_num] = winners
        results['rounds'][round_name] = [
            {'team': w.name, 'seed': w.seed} for w in winners
        ]
        
        # Update regions
        if round_num <= 4:
            for region in bracket.regions:
                bracket.regions[region] = [
                    t for t in bracket.regions[region] if t in winners
                ]
    
    results['champion'] = winners[0].name if winners else None
    results['final_four'] = [
        p['winner'] for p in results['predictions']
        if p['round'] == 'Elite 8'
    ]
    
    return results
```

## Parallel Simulation

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

def parallel_simulation(predictor, initial_bracket: BracketState,
                        n_simulations: int = 10000,
                        n_workers: int = None) -> Dict:
    """Run simulations in parallel for speed."""
    
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    sims_per_worker = n_simulations // n_workers
    
    def run_batch(seed: int) -> Dict:
        """Run a batch of simulations."""
        simulator = BracketSimulator(predictor, seed=seed)
        return simulator.run_simulations(initial_bracket, sims_per_worker)
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        seeds = list(range(n_workers))
        results = list(executor.map(run_batch, seeds))
    
    # Aggregate results
    combined = aggregate_simulation_results(results)
    combined['n_simulations'] = n_simulations
    
    return combined


def aggregate_simulation_results(results: List[Dict]) -> Dict:
    """Combine results from multiple simulation batches."""
    
    total_sims = sum(r['n_simulations'] for r in results)
    
    # Combine championship counts
    championship = {}
    for r in results:
        for team, prob in r['championship_distribution'].items():
            count = prob * r['n_simulations']
            championship[team] = championship.get(team, 0) + count
    
    # Convert back to probabilities
    championship_probs = {k: v / total_sims for k, v in championship.items()}
    
    # Combine team probabilities
    team_probs = {}
    for r in results:
        weight = r['n_simulations'] / total_sims
        for team_id, probs in r['team_probabilities'].items():
            if team_id not in team_probs:
                team_probs[team_id] = {
                    'name': probs['name'],
                    'seed': probs['seed'],
                    'region': probs['region']
                }
                for key in ['round_32_prob', 'sweet_16_prob', 'elite_8_prob',
                           'final_four_prob', 'finals_prob', 'champion_prob']:
                    team_probs[team_id][key] = 0
            
            for key in ['round_32_prob', 'sweet_16_prob', 'elite_8_prob',
                       'final_four_prob', 'finals_prob', 'champion_prob']:
                team_probs[team_id][key] += probs[key] * weight
    
    return {
        'team_probabilities': team_probs,
        'championship_distribution': championship_probs,
        'most_likely_champion': max(championship_probs, key=championship_probs.get)
    }
```

## Simulation Output

```python
def format_simulation_results(results: Dict) -> str:
    """Format simulation results for display."""
    
    output = []
    output.append("=" * 60)
    output.append("BRACKET SIMULATION RESULTS")
    output.append(f"Based on {results['n_simulations']:,} simulations")
    output.append("=" * 60)
    
    # Championship probabilities (top 10)
    output.append("\n🏆 CHAMPIONSHIP PROBABILITIES:")
    sorted_champs = sorted(
        results['championship_distribution'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for i, (team, prob) in enumerate(sorted_champs, 1):
        output.append(f"  {i:2}. {team:25} {prob:.1%}")
    
    # Final Four probabilities (top 16)
    output.append("\n🏀 FINAL FOUR PROBABILITIES:")
    sorted_ff = sorted(
        [(t, p['final_four_prob']) for t, p in results['team_probabilities'].items()],
        key=lambda x: x[1],
        reverse=True
    )[:16]
    
    for i, (team, prob) in enumerate(sorted_ff, 1):
        output.append(f"  {i:2}. {team:25} {prob:.1%}")
    
    return "\n".join(output)


def export_simulation_results(results: Dict, filepath: str):
    """Export results to JSON for visualization."""
    import json
    
    export_data = {
        'summary': {
            'n_simulations': results['n_simulations'],
            'most_likely_champion': results['most_likely_champion'],
            'top_championship_contenders': dict(sorted(
                results['championship_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:10])
        },
        'all_teams': results['team_probabilities'],
        'championship_distribution': results['championship_distribution']
    }
    
    with open(filepath, 'w') as f:
        json.dump(export_data, f, indent=2)
```

## Next Steps

1. Integrate with existing prediction models
2. Build bracket state from ESPN data
3. See `roadmap-bracket-visualization.md` for displaying results
4. See `roadmap-upset-detection.md` for upset analysis
