# Phase 2: Advanced Feature Engineering

*Based on the Claude Opus 4.6 Bracket Codex Research Dossier — Theories 4–6, Improvements #3–6, #9–10*

## Overview

Phase 2 adds five new feature families to the model pipeline: matchup-specific adjustments, three-point variance modeling, coaching experience, conference tournament fatigue, and travel/venue effects. Each feature is designed as an additive column in the training data so the existing XGBoost / RF / LR ensemble can learn appropriate weights without architectural changes.

---

## 2.1 Matchup-Specific Adjustments (The "Kill Shot" Theory)

### Why

The dossier's Theory #4 ("Kill Shot — Style Collision Matrix") observes that certain offensive profiles systematically exploit specific defensive weaknesses. For example, a team that lives on the 3-point line (+3PT rate top-20 %) against a defense that allows above-average 3PT percentage creates a mismatch the efficiency differentials alone don't capture. The current model uses `off_eff_diff` and `def_eff_diff` which flatten all scoring into per-possession averages, masking *how* teams score and defend.

### Expected Effect

- **Moneyline model**: Adding matchup interaction features should improve classification accuracy by 1–2 % on tournament games, particularly for mid-seed matchups (5-vs-12, 6-vs-11) where style clashes drive upsets.
- **Spread model**: Matchup features refine point-spread predictions in games where one team's primary offensive weapon attacks the other's specific weakness.
- **Upset detection**: The "Kill Shot" feature directly identifies which underdogs have stylistic advantages, reducing false negatives in upset prediction.

### Implementation

Add to `features.py`:

```python
# --- Research Dossier §8.4 — Kill Shot Matchup Features -------------------

def calculate_matchup_features(team1_stats: dict, team2_stats: dict) -> dict:
    """Compute matchup-specific interaction features.

    Captures how team1's offensive strengths interact with team2's
    defensive weaknesses (and vice versa).

    Args:
        team1_stats: Dict with keys like 'three_pct', 'efg_pct', 'to_rate',
                     'orb_pct', 'ft_rate', 'opp_three_pct', 'opp_efg_pct', etc.
        team2_stats: Same structure for the opponent.

    Returns:
        Dict of matchup interaction features.
    """
    # Team 1 3PT offense vs Team 2 3PT defense
    t1_three = _get_eff(team1_stats, 'three_pct', default=0.33)
    t2_opp_three = _get_eff(team2_stats, 'opp_three_pct', default=0.33)

    # Team 2 3PT offense vs Team 1 3PT defense
    t2_three = _get_eff(team2_stats, 'three_pct', default=0.33)
    t1_opp_three = _get_eff(team1_stats, 'opp_three_pct', default=0.33)

    # Turnover forcing: team defense TO rate vs opponent's care with the ball
    t1_opp_to = _get_eff(team1_stats, 'opp_to_rate', default=0.18)
    t2_to = _get_eff(team2_stats, 'to_rate', default=0.18)
    t2_opp_to = _get_eff(team2_stats, 'opp_to_rate', default=0.18)
    t1_to = _get_eff(team1_stats, 'to_rate', default=0.18)

    # Rebounding mismatch
    t1_orb = _get_eff(team1_stats, 'orb_pct', default=0.28)
    t2_drb = _get_eff(team2_stats, 'drb_pct', default=0.72)
    t2_orb = _get_eff(team2_stats, 'orb_pct', default=0.28)
    t1_drb = _get_eff(team1_stats, 'drb_pct', default=0.72)

    return {
        # Positive = team1 has the 3PT mismatch advantage
        'three_pt_mismatch': (t1_three - t2_opp_three) - (t2_three - t1_opp_three),
        # Positive = team1 forces more turnovers relative to their own
        'turnover_mismatch': (t1_opp_to - t2_to) - (t2_opp_to - t1_to),
        # Positive = team1 has the rebounding edge
        'rebound_mismatch': (t1_orb - (1 - t2_drb)) - (t2_orb - (1 - t1_drb)),
        # Free-throw rate differential (fouling pressure)
        'ft_rate_mismatch': (
            _get_eff(team1_stats, 'ft_rate', default=0.30)
            - _get_eff(team2_stats, 'ft_rate', default=0.30)
        ),
    }
```

Add columns to `feature_engineering.py` training data builder:

```python
# After computing efficiency diffs for each game row:
matchup_feats = calculate_matchup_features(home_stats, away_stats)
for k, v in matchup_feats.items():
    row[f'matchup_{k}'] = v
```

---

## 2.2 Three-Point Variance Modeling

### Why

The dossier's Improvement #5 ("3PT Variance Modeling") and Theory #6 ("VIG — Variance Is the Great Equalizer") highlight that high-variance three-point shooting is the single largest source of tournament upsets. A team shooting 38 % from three has the same average as one shooting 33 % but with higher game-to-game variance — the latter is more likely to both dramatically over-perform (upset wins) and under-perform (upset losses). The current model only uses `three_pct` (mean), ignoring variance.

### Expected Effect

- **Upset detection**: Variance features should improve upset AUC by 3–5 %. High-variance underdogs are more likely to have a "hot" shooting game.
- **Spread model**: Three-point variance widens the confidence interval on spread predictions, which should improve ATS (against the spread) identification.
- **Over/under**: High combined variance teams generate wider score distributions, directly affecting total predictions.

### Implementation

Add to `features.py`:

```python
# --- Research Dossier §8.6 — Three-Point Variance (VIG) -------------------

def calculate_three_pt_variance_features(
    team1_stats: dict, team2_stats: dict
) -> dict:
    """Compute three-point variance features for upset potential.

    Uses game-to-game standard deviation of 3PT% (if available) or
    estimates it from season averages using the empirical relationship
    σ(3PT%) ≈ 0.08 + 0.12 * |3PT% − league_avg|.

    Args:
        team1_stats: Must contain 'three_pct' and optionally 'three_pct_std'.
        team2_stats: Same structure.

    Returns:
        Dict of variance features.
    """
    LEAGUE_AVG_3PT = 0.338  # D-I average 2024-25

    def _est_std(stats: dict) -> float:
        """Estimate 3PT% game-to-game std dev."""
        if 'three_pct_std' in stats and stats['three_pct_std']:
            return float(stats['three_pct_std'])
        pct = _get_eff(stats, 'three_pct', default=LEAGUE_AVG_3PT)
        return 0.08 + 0.12 * abs(pct - LEAGUE_AVG_3PT)

    t1_std = _est_std(team1_stats)
    t2_std = _est_std(team2_stats)
    t1_pct = _get_eff(team1_stats, 'three_pct', default=LEAGUE_AVG_3PT)
    t2_pct = _get_eff(team2_stats, 'three_pct', default=LEAGUE_AVG_3PT)

    return {
        # Higher = more combined variance in the game = more upset-prone
        'combined_3pt_variance': t1_std + t2_std,
        # Team 1's relative variance advantage (higher variance = more ability to "get hot")
        'three_pt_var_diff': t1_std - t2_std,
        # Coefficient of variation: variance relative to mean
        'three_pt_cv_diff': (
            (t1_std / t1_pct if t1_pct > 0 else 0)
            - (t2_std / t2_pct if t2_pct > 0 else 0)
        ),
    }
```

---

## 2.3 Coaching Experience Multiplier

### Why

Improvement #10 in the dossier ("Coaching Experience Multiplier") cites that coaches with 5+ tournament appearances win 8 % more games than first-timers in equivalent seed matchups. Tournament coaching adjustments (timeout pacing, defensive sets in final 4 minutes, foul management) are not captured by regular-season efficiency metrics.

### Expected Effect

- **Late-round predictions**: Coaching experience matters most in close tournament games. Adding this feature should improve Elite 8 and Final Four prediction accuracy by 2–3 %.
- **Bracket simulation**: Experienced coaches are less likely to suffer first-round upsets, which compounding through 6 rounds significantly alters champion distribution.

### Implementation

Create `data_files/coaching_experience.json` (manually maintained or scraped):

```json
{
    "Duke": {"coach": "Jon Scheyer", "tourney_appearances": 3, "final_fours": 0},
    "Kansas": {"coach": "Bill Self", "tourney_appearances": 22, "final_fours": 5},
    "Gonzaga": {"coach": "Mark Few", "tourney_appearances": 25, "final_fours": 2},
    "Connecticut": {"coach": "Dan Hurley", "tourney_appearances": 6, "final_fours": 3}
}
```

Add to `features.py`:

```python
# --- Research Dossier Improvement #10 — Coaching Experience ----------------

import json
from pathlib import Path

_COACHING_CACHE = None

def _load_coaching_data() -> dict:
    global _COACHING_CACHE
    if _COACHING_CACHE is None:
        path = Path('data_files/coaching_experience.json')
        if path.exists():
            with open(path) as f:
                _COACHING_CACHE = json.load(f)
        else:
            _COACHING_CACHE = {}
    return _COACHING_CACHE


def calculate_coaching_features(team1_name: str, team2_name: str) -> dict:
    """Coaching tournament experience differential.

    Returns:
        Dict with coaching_exp_diff (appearances) and coaching_ff_diff (final fours).
    """
    data = _load_coaching_data()
    t1 = data.get(team1_name, {})
    t2 = data.get(team2_name, {})

    return {
        'coaching_exp_diff': t1.get('tourney_appearances', 0) - t2.get('tourney_appearances', 0),
        'coaching_ff_diff': t1.get('final_fours', 0) - t2.get('final_fours', 0),
    }
```

---

## 2.4 Conference Tournament Fatigue

### Why

Improvement #9 ("Conference Tournament Fatigue") notes that teams playing 4+ conference tournament games in the week before the NCAA tournament underperform their ratings by 1.5–2 points in Round 1. The current model has no awareness of pre-tournament game load.

### Expected Effect

- **First-round spread model**: Fatigue features should reduce first-round spread MAE by 0.5–1.0 points for affected teams (typically auto-bid mid-major champions who played 4–5 games in 5 days).
- **Upset detection**: Fatigued favorites are more vulnerable to upsets; flagging these games improves the upset model's recall.

### Implementation

Add to `features.py`:

```python
# --- Research Dossier Improvement #9 — Conference Tournament Fatigue -------

FATIGUE_THRESHOLD_GAMES = 3  # 4+ games triggers fatigue adjustment
FATIGUE_PPG_PENALTY = 1.5    # Historical underperformance per extra game


def calculate_fatigue_features(
    team1_conf_tourney_games: int,
    team2_conf_tourney_games: int,
) -> dict:
    """Conference tournament fatigue differential.

    Args:
        team1_conf_tourney_games: Number of conf tournament games played by team 1.
        team2_conf_tourney_games: Number of conf tournament games played by team 2.

    Returns:
        Dict with fatigue features.
    """
    t1_fatigue = max(0, team1_conf_tourney_games - FATIGUE_THRESHOLD_GAMES)
    t2_fatigue = max(0, team2_conf_tourney_games - FATIGUE_THRESHOLD_GAMES)

    return {
        'fatigue_diff': t1_fatigue - t2_fatigue,
        'combined_fatigue': t1_fatigue + t2_fatigue,
        'fatigue_spread_adj': (t2_fatigue - t1_fatigue) * FATIGUE_PPG_PENALTY,
    }
```

---

## 2.5 Travel & Venue Effects

### Why

Improvement #6 ("Travel / Venue Effects") documents that teams traveling 500+ miles to tournament sites underperform by 0.8 points on average. Teams playing within 200 miles of their campus get a de facto home-court bump. The current model has no venue or location awareness.

### Expected Effect

- **Spread model**: Travel distance adds a 0.5–1.5 point refinement for games at neutral sites that are geographically closer to one team.
- **Bracket simulation**: Venue effects compound through multiple rounds — a team playing near home in the first two rounds gains a persistent advantage.

### Implementation

Add to `features.py`:

```python
# --- Research Dossier Improvement #6 — Travel/Venue Effects ----------------

from math import radians, sin, cos, sqrt, atan2

# Approximate campus coordinates for major programs (extend as needed).
# In production, load from a JSON file.
CAMPUS_COORDS = {}  # Populated from data_files/campus_locations.json

VENUE_COORDS = {
    # 2026 tournament sites — update each year
    'Dayton': (39.76, -84.19),
    'Pittsburgh': (40.44, -79.99),
    'Indianapolis': (39.77, -86.16),
    'Memphis': (35.15, -90.05),
    'Dallas': (32.78, -96.80),
    'San Antonio': (29.42, -98.49),
}


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles between two lat/lon points."""
    R = 3958.8  # Earth radius in miles
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def calculate_travel_features(
    team1_name: str,
    team2_name: str,
    venue_name: str,
    campus_coords: dict = None,
    venue_coords: dict = None,
) -> dict:
    """Compute travel distance differential feature.

    Returns:
        Dict with travel_diff_miles (negative = team1 closer to venue)
        and venue_advantage (estimated point adjustment for team1).
    """
    campuses = campus_coords or CAMPUS_COORDS
    venues = venue_coords or VENUE_COORDS

    t1_loc = campuses.get(team1_name)
    t2_loc = campuses.get(team2_name)
    v_loc = venues.get(venue_name)

    if not (t1_loc and t2_loc and v_loc):
        return {'travel_diff_miles': 0.0, 'venue_advantage': 0.0}

    d1 = _haversine_miles(t1_loc[0], t1_loc[1], v_loc[0], v_loc[1])
    d2 = _haversine_miles(t2_loc[0], t2_loc[1], v_loc[0], v_loc[1])

    diff = d1 - d2  # Negative means team1 is closer

    # ~0.8 points per 500-mile disadvantage (dossier estimate)
    advantage = -diff * (0.8 / 500.0)

    return {
        'travel_diff_miles': diff,
        'venue_advantage': advantage,
    }
```

Create `data_files/campus_locations.json` with lat/lon for D-I programs.

---

## Integration into Training Pipeline

All new features from this phase are computed as additional columns in `feature_engineering.py`. The model training code in `model_training.py` already dynamically selects columns by prefix (`spread_*`, `total_*`), so new features will be picked up automatically if named with the appropriate prefix:

```python
# In feature_engineering.py, after existing feature computation:

# Phase 2 features — matchup, variance, coaching, fatigue, travel
matchup_feats = calculate_matchup_features(home_stats, away_stats)
for k, v in matchup_feats.items():
    row[f'spread_{k}'] = v    # Also useful for spread prediction
    row[f'total_{k}'] = v     # Some matchup features affect totals too

variance_feats = calculate_three_pt_variance_features(home_stats, away_stats)
for k, v in variance_feats.items():
    row[f'spread_{k}'] = v
    row[f'total_{k}'] = v

coaching_feats = calculate_coaching_features(home_name, away_name)
for k, v in coaching_feats.items():
    row[f'spread_{k}'] = v

fatigue_feats = calculate_fatigue_features(
    home_conf_tourney_games, away_conf_tourney_games
)
for k, v in fatigue_feats.items():
    row[f'spread_{k}'] = v

travel_feats = calculate_travel_features(home_name, away_name, venue)
for k, v in travel_feats.items():
    row[f'spread_{k}'] = v
```

---

## Data Requirements

| Feature | Data Source | Available Today? | Action Needed |
|---------|-----------|-----------------|---------------|
| Matchup stats | ESPN team stats (four factors) | ✅ Yes | Already in `feature_engineering.py` |
| 3PT variance | Game-by-game logs | ⚠️ Partial | Need game logs or use estimation formula |
| Coaching experience | Manual / Sports Reference | ❌ No | Create `coaching_experience.json` (~68 teams) |
| Conf tourney games | ESPN schedule API | ✅ Yes | Count games in March pre-Selection Sunday |
| Campus coordinates | Geocoding / manual | ❌ No | Create `campus_locations.json` |

---

## Files Modified / Created

| File | Change |
|------|--------|
| `features.py` | Add 5 new feature functions |
| `feature_engineering.py` | Wire new features into training data builder |
| `data_files/coaching_experience.json` | **New** — coaching tournament experience data |
| `data_files/campus_locations.json` | **New** — lat/lon for D-I campuses |
