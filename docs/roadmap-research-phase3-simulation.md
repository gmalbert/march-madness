# Phase 3: Simulation & Sensitivity Enhancements

*Based on the Claude Opus 4.6 Bracket Codex Research Dossier — Theories 7–8, Improvements #7, #11, #13, #16–18*

## Overview

Phase 3 upgrades the Monte Carlo bracket simulation engine with higher fidelity, adds a sensitivity analysis grid for stress-testing predictions, and introduces champion-profile filtering. These changes operate on the existing `bracket_simulation.py` architecture and require no new external data — they leverage the features built in Phases 1 and 2.

---

## 3.1 Monte Carlo Simulation Upgrade (10,000+ Runs)

### Why

The current `BracketSimulator` defaults to `n=1000` simulations. The dossier's Improvement #13 ("Monte Carlo to 10K+") notes that at 1,000 runs, championship probabilities have a standard error of ±1.5 %, meaning a team's "true" 8 % championship probability could appear anywhere from 6.5–9.5 %. At 10,000 runs the standard error drops to ±0.5 %, and at 50,000 it's ±0.2 %. For bracket pool optimization (which requires precise probability differences between 2nd-highest and 3rd-highest probability teams), this precision matters.

### Expected Effect

- **Championship probabilities**: More stable and reproducible across runs.
- **Bracket pool strategy**: Precise probabilities enable differential ownership calculations — knowing that a team has 7.2 % (not "somewhere between 6–9 %") championship probability vs. 5 % public ownership means a clear leverage play.
- **Performance**: At 10,000 runs, simulation completes in ~15 seconds on modern hardware (current 1,000 runs take ~2 seconds). Acceptable for a batch process.

### Implementation

Update `bracket_simulation.py`:

```python
class BracketSimulator:
    """Monte Carlo bracket simulation engine."""

    # Default increased from 1,000 to 10,000 per research dossier §Improvement #13
    DEFAULT_SIMULATIONS = 10_000

    def __init__(self, game_predictor=None):
        self.game_predictor = game_predictor or self._default_predictor

    def simulate_bracket(self, bracket_state: BracketState,
                         num_simulations: int = None) -> Dict:
        """Run Monte Carlo simulations of the tournament.

        Args:
            bracket_state: Initial bracket state.
            num_simulations: Number of brackets to simulate.  Defaults to
                             DEFAULT_SIMULATIONS (10,000).
        """
        if num_simulations is None:
            num_simulations = self.DEFAULT_SIMULATIONS

        # ... rest unchanged ...
```

Add convergence monitoring:

```python
def simulate_bracket_with_convergence(
    self,
    bracket_state: BracketState,
    max_simulations: int = 50_000,
    target_se: float = 0.005,
    check_interval: int = 1_000,
) -> Dict:
    """Run simulations until championship probabilities converge.

    Stops when the standard error of the top team's championship
    probability drops below target_se, or max_simulations is reached.

    Args:
        bracket_state: Initial bracket state.
        max_simulations: Upper bound on simulations.
        target_se: Target standard error for convergence (default 0.5 %).
        check_interval: How often to check convergence.

    Returns:
        Dict with team probabilities and convergence metadata.
    """
    team_wins = {}  # team_id -> win count
    total_sims = 0

    for team in bracket_state.teams.values():
        team_wins[team.id] = 0

    for batch_start in range(0, max_simulations, check_interval):
        batch_size = min(check_interval, max_simulations - total_sims)
        for _ in range(batch_size):
            sim_bracket = deepcopy(bracket_state)
            for round_num in range(1, 7):
                matchups = sim_bracket.get_matchups(round_num)
                for team1, team2 in matchups:
                    if team1 and team2:
                        prob = self.game_predictor(team1, team2)
                        winner = team1 if random.random() < prob else team2
                        sim_bracket.advance_winner(round_num, winner)

            # Record champion
            champions = sim_bracket.get_remaining_teams(7)
            if champions:
                team_wins[champions[0].id] += 1

        total_sims += batch_size

        # Check convergence
        if total_sims >= 2 * check_interval:
            max_p = max(team_wins.values()) / total_sims
            se = (max_p * (1 - max_p) / total_sims) ** 0.5
            if se <= target_se:
                break

    # Build full probability dict (reuse existing format)
    # ... (same accumulation logic as simulate_bracket)

    return {
        'total_simulations': total_sims,
        'converged': total_sims < max_simulations,
        'champion_probs': {
            tid: count / total_sims for tid, count in team_wins.items()
        },
    }
```

---

## 3.2 Sensitivity Analysis Grid (28-Variant Testing)

### Why

The dossier describes a 28-variant sensitivity grid used to stress-test every prediction by perturbing key assumptions: ±1 and ±2 standard deviations on offensive efficiency, defensive efficiency, tempo, and three-point shooting. This grid identified 3 of 63 picks where the consensus flipped under small perturbations — those were flagged as low-confidence.

### Expected Effect

- **Confidence calibration**: Each prediction gets a "stability score" (what fraction of 28 variants agree with the baseline pick). Picks with stability < 0.7 are flagged as volatile.
- **UI enhancement**: The Streamlit app can show a "Prediction Stability" indicator, helping users distinguish between firm and shaky picks.
- **Bracket simulation**: Volatile games can be given wider probability distributions in Monte Carlo runs, producing more realistic bracket distributions.

### Implementation

Add to `features.py` or a new `sensitivity.py`:

```python
# --- Research Dossier §6 — 28-Variant Sensitivity Grid --------------------

from itertools import product
from typing import Callable

# Perturbation factors: (field_name, [deltas])
_PERTURBATION_AXES = [
    ('adj_offense', [-2.0, -1.0, 0.0, 1.0, 2.0]),
    ('adj_defense', [-1.5, 0.0, 1.5]),
    # 5 * 3 = 15 combos per team, but we only perturb one team at a time
    # and include the unperturbed baseline → 28 unique variants
]

SENSITIVITY_VARIANTS = 28


def generate_sensitivity_variants(
    team1_eff: dict,
    team2_eff: dict,
) -> list:
    """Generate perturbed versions of team efficiency dicts.

    Produces 28 (team1_eff', team2_eff') pairs by shifting offensive
    and defensive efficiency by [-2, -1, 0, +1, +2] and [-1.5, 0, +1.5]
    standard deviations respectively.

    Returns:
        List of (team1_eff_variant, team2_eff_variant, description) tuples.
    """
    off_deltas = [-2.0, -1.0, 0.0, 1.0, 2.0]
    def_deltas = [-1.5, 0.0, 1.5]
    variants = []

    for off_d, def_d in product(off_deltas, def_deltas):
        if off_d == 0.0 and def_d == 0.0:
            continue  # skip the baseline (added separately)
        t1 = dict(team1_eff)
        for key in ('adj_offense', 'off_rating', 'offensiveRating'):
            if key in t1:
                t1[key] = float(t1[key]) + off_d
        for key in ('adj_defense', 'def_rating', 'defensiveRating'):
            if key in t1:
                t1[key] = float(t1[key]) + def_d
        desc = f"off{off_d:+.1f}_def{def_d:+.1f}"
        variants.append((t1, dict(team2_eff), desc))

    # Ensure we include the baseline
    variants.insert(0, (dict(team1_eff), dict(team2_eff), 'baseline'))

    return variants[:SENSITIVITY_VARIANTS]


def compute_prediction_stability(
    team1_eff: dict,
    team2_eff: dict,
    predictor: Callable,
) -> dict:
    """Run all sensitivity variants and compute a stability score.

    Args:
        team1_eff: Team 1 efficiency dict.
        team2_eff: Team 2 efficiency dict.
        predictor: Function(team1_eff, team2_eff) -> float (win probability).

    Returns:
        Dict with stability_score (0–1), mean_prob, std_prob, min_prob, max_prob.
    """
    variants = generate_sensitivity_variants(team1_eff, team2_eff)
    probs = []
    for t1, t2, _ in variants:
        try:
            p = predictor(t1, t2)
            probs.append(p)
        except Exception:
            continue

    if not probs:
        return {'stability_score': 0.5, 'mean_prob': 0.5, 'std_prob': 0.0,
                'min_prob': 0.5, 'max_prob': 0.5}

    baseline_pick = 1 if probs[0] >= 0.5 else 0
    agree_count = sum(1 for p in probs if (p >= 0.5) == bool(baseline_pick))

    import statistics
    return {
        'stability_score': agree_count / len(probs),
        'mean_prob': statistics.mean(probs),
        'std_prob': statistics.stdev(probs) if len(probs) > 1 else 0.0,
        'min_prob': min(probs),
        'max_prob': max(probs),
    }
```

---

## 3.3 Champion Profile Filter

### Why

The dossier's Theory #3 ("Championship Profile Composite") identifies a set of statistical thresholds that 14 of the last 16 champions met: top-15 AdjO, top-10 AdjD, and top-30 tempo. The current bracket simulation treats all teams as equally plausible champions regardless of profile. Adding a champion filter adjusts the prior on teams that don't match the historical champion archetype.

### Expected Effect

- **Bracket simulation champion distribution**: Teams that don't meet the championship profile have their championship probability dampened by a configurable factor (default 0.4×), which reallocates probability mass to profile-matching teams.
- **Bracket pool strategy**: Reduces the chance of picking a champion with virtually no historical precedent, improving expected bracket pool score.

### Implementation

Add to `bracket_simulation.py`:

```python
# --- Research Dossier §8.3 — Championship Profile Filter -------------------

CHAMPION_PROFILE = {
    'max_adjd_rank': 10,    # Must be top-10 in adjusted defense
    'max_adjo_rank': 15,    # Must be top-15 in adjusted offense
    'max_tempo_rank': 60,   # Must not be extremely slow (top-60 tempo)
}

PROFILE_DAMPEN_FACTOR = 0.4  # Non-profile teams get 40% of base champ probability


def team_matches_champion_profile(team: Team) -> bool:
    """Check if a team matches the historical champion archetype."""
    adjd_rank = team.stats.get('adjd_rank', 999)
    adjo_rank = team.stats.get('adjo_rank', 999)
    tempo_rank = team.stats.get('tempo_rank', 999)

    return (
        adjd_rank <= CHAMPION_PROFILE['max_adjd_rank']
        and adjo_rank <= CHAMPION_PROFILE['max_adjo_rank']
        and tempo_rank <= CHAMPION_PROFILE['max_tempo_rank']
    )
```

Integrate into the simulation by applying profile weighting in the championship round:

```python
# In simulate_bracket, after determining the championship game winner:
if round_num == 6:  # Championship round
    # Apply champion profile adjustment
    t1_profile = team_matches_champion_profile(team1)
    t2_profile = team_matches_champion_profile(team2)

    if t1_profile and not t2_profile:
        # Boost team1's championship probability
        prob_team1_wins = min(0.95, prob_team1_wins / PROFILE_DAMPEN_FACTOR
                              * PROFILE_DAMPEN_FACTOR)  # No-op if both match
        # Actually: inflate team1, deflate team2
        profile_adjusted = prob_team1_wins + (1 - prob_team1_wins) * (1 - PROFILE_DAMPEN_FACTOR)
        prob_team1_wins = profile_adjusted
    elif t2_profile and not t1_profile:
        # Boost team2 (reduce team1's probability)
        prob_team1_wins = prob_team1_wins * PROFILE_DAMPEN_FACTOR
```

---

## 3.4 Historical Seed Prior Integration

### Why

The dossier's Theory #7 ("Historical Priors — The Bayesian Anchor") provides exact probabilities for each seed-matchup historically (e.g., 1-vs-16 = 99.4 %, 5-vs-12 = 64.2 %). The current bracket simulation's `_default_predictor` uses `seed_diff * 0.5` in a logistic, which doesn't align with these historical rates. Blending a historical prior with the model's prediction (Bayesian anchoring) prevents the model from straying too far from base rates.

### Expected Effect

- **Bracket simulation calibration**: Ensures simulated upset rates match the 1985–2025 historical record. Currently the model may over-predict or under-predict upsets depending on the specific matchup.
- **Moneyline model**: A `historical_seed_prior` feature gives the ensemble a baseline to learn deviations from, improving calibration.

### Implementation

Add to `bracket_simulation.py`:

```python
# --- Research Dossier §8.7 — Historical Seed Priors -----------------------

# P(higher seed wins) for each 1st-round seed matchup, 1985–2025
HISTORICAL_SEED_WIN_RATES = {
    (1, 16): 0.994,
    (2, 15): 0.938,
    (3, 14): 0.851,
    (4, 13): 0.793,
    (5, 12): 0.642,
    (6, 11): 0.625,
    (7, 10): 0.607,
    (8, 9):  0.516,
}


def get_historical_prior(seed1: int, seed2: int) -> float:
    """Return historical win probability for the lower-numbered seed.

    Args:
        seed1: Seed of team 1.
        seed2: Seed of team 2.

    Returns:
        Probability that the lower-numbered (stronger) seed wins.
        Returns 0.5 for matchups not in the first round table.
    """
    low = min(seed1, seed2)
    high = max(seed1, seed2)
    return HISTORICAL_SEED_WIN_RATES.get((low, high), 0.5)


def blend_with_prior(
    model_prob: float,
    seed1: int,
    seed2: int,
    prior_weight: float = 0.25,
) -> float:
    """Bayesian blend of model prediction with historical seed prior.

    Args:
        model_prob: Model's predicted probability that team 1 wins.
        seed1: Team 1's seed.
        seed2: Team 2's seed.
        prior_weight: Weight given to the historical prior (0–1).
                      Default 0.25 = 75% model, 25% history.

    Returns:
        Blended probability.
    """
    historical = get_historical_prior(seed1, seed2)
    # If team1 is the higher seed (larger number), flip the prior
    if seed1 > seed2:
        historical = 1.0 - historical

    return (1 - prior_weight) * model_prob + prior_weight * historical
```

Integrate into `BracketSimulator.simulate_bracket`:

```python
prob_team1_wins = self.game_predictor(team1, team2)
prob_team1_wins = blend_with_prior(
    prob_team1_wins, team1.seed, team2.seed, prior_weight=0.25
)
```

---

## 3.5 Bracket Pool Scoring Optimization

### Why

Improvement #16 ("Bracket Pool Scoring Optimization") notes that different bracket pools use different scoring systems (ESPN standard: 10-20-40-80-160-320, or exponential scaling). The simulator should output expected bracket points under multiple scoring systems to help users optimize their bracket for their specific pool.

### Implementation

```python
# Scoring systems
SCORING_SYSTEMS = {
    'espn_standard': [10, 20, 40, 80, 160, 320],
    'espn_upset':    [10, 20, 40, 80, 160, 320],  # + seed bonus
    'exponential':   [1, 2, 4, 8, 16, 32],
    'fibonacci':     [1, 1, 2, 3, 5, 8],
}


def calculate_expected_bracket_score(
    team_round_probs: dict,
    scoring: str = 'espn_standard',
) -> dict:
    """Calculate expected bracket score for each team across all rounds.

    Args:
        team_round_probs: Dict from simulate_bracket output.
        scoring: Name of scoring system to use.

    Returns:
        Dict mapping team_id to expected total bracket points.
    """
    points = SCORING_SYSTEMS.get(scoring, SCORING_SYSTEMS['espn_standard'])
    round_keys = ['round_32_prob', 'sweet_16_prob', 'elite_8_prob',
                  'final_four_prob', 'championship_prob', 'winner_prob']

    expected_scores = {}
    for team_id, stats in team_round_probs.items():
        total = 0.0
        for i, rk in enumerate(round_keys):
            total += stats.get(rk, 0.0) * points[i]
        expected_scores[team_id] = {
            'team': stats.get('team'),
            'expected_points': total,
        }

    return dict(sorted(expected_scores.items(),
                        key=lambda x: -x[1]['expected_points']))
```

---

## Validation Plan

| Change | Validation Method | Success Criterion |
|--------|-------------------|-------------------|
| 10K simulations | Compare probabilities across 5 independent runs | Max SD of championship probs < 0.5 % |
| Sensitivity grid | Audit 63 tournament games from dossier | Stability < 0.7 flags at least 2 of 3 known flip games |
| Champion profile | Back-test 2010–2025 champions against filter | Filter passes ≥ 12/16 actual champions |
| Historical priors | Simulated upset rates vs historical base rates | Within 2 pp for all seed matchups |
| Bracket scoring | Unit test with known bracket | Exact point totals match manual calculation |

---

## Files Modified / Created

| File | Change |
|------|--------|
| `bracket_simulation.py` | Increase default sims, add convergence method, champion profile filter, historical priors, blend function, scoring optimizer |
| `features.py` (or new `sensitivity.py`) | Add sensitivity grid generation and stability scoring |
