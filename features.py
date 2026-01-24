"""Feature engineering helpers for betting models.

Implements functions from docs/roadmap-betting-features.md with safe
lookups and reasonable defaults so they can be used across the codebase.
"""
from typing import Dict, List, Any


def _get_eff(e: Dict[str, Any], *keys, default=0.0) -> float:
    for k in keys:
        if isinstance(e, dict) and k in e and e[k] is not None:
            return float(e[k])
    return float(default)


def calculate_efficiency_differential(team1_eff: dict, team2_eff: dict) -> dict:
    """Calculate efficiency differentials between two teams."""
    t1_off = _get_eff(team1_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t1_def = _get_eff(team1_eff, 'adj_defense', 'def_rating', 'defensiveRating')
    t2_off = _get_eff(team2_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t2_def = _get_eff(team2_eff, 'adj_defense', 'def_rating', 'defensiveRating')

    return {
        "off_eff_diff": t1_off - t2_off,
        "def_eff_diff": t1_def - t2_def,
        "net_eff_diff": (t1_off - t1_def) - (t2_off - t2_def),
    }


def project_game_total(team1_eff: dict, team2_eff: dict) -> float:
    """Project total points using tempo and efficiency.

    Formula: (Team1_OE + Team2_OE) / 2 * Avg_Tempo / 100  (scaled to game)
    """
    tempo1 = _get_eff(team1_eff, 'tempo', 'pace')
    tempo2 = _get_eff(team2_eff, 'tempo', 'pace')
    avg_tempo = (tempo1 + tempo2) / 2 or 70

    t1_off = _get_eff(team1_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t2_off = _get_eff(team2_eff, 'adj_offense', 'off_rating', 'offensiveRating')
    t1_def = _get_eff(team1_eff, 'adj_defense', 'def_rating', 'defensiveRating')
    t2_def = _get_eff(team2_eff, 'adj_defense', 'def_rating', 'defensiveRating')

    avg_off_eff = (t1_off + t2_off) / 2
    avg_def_eff = (t1_def + t2_def) / 2

    expected_eff = (avg_off_eff + avg_def_eff) / 2

    # Scale to game possessions (expected_eff is points per 100 possessions)
    projected_total = expected_eff * (avg_tempo / 100) * 2
    return float(projected_total)


def calculate_spread_features(team1: dict, team2: dict) -> dict:
    """Features for predicting point spread."""
    return {
        "net_rating_diff": _get_eff(team1, 'net_rating') - _get_eff(team2, 'net_rating'),
        "off_rating_diff": _get_eff(team1, 'off_rating') - _get_eff(team2, 'off_rating'),
        "def_rating_diff": _get_eff(team1, 'def_rating') - _get_eff(team2, 'def_rating'),
        "ppg_diff": _get_eff(team1, 'ppg') - _get_eff(team2, 'ppg'),
        "opp_ppg_diff": _get_eff(team1, 'opp_ppg') - _get_eff(team2, 'opp_ppg'),
        "margin_diff": _get_eff(team1, 'avg_margin') - _get_eff(team2, 'avg_margin'),
        "seed_diff": _get_eff(team1, 'seed') - _get_eff(team2, 'seed'),
        "efg_diff": _get_eff(team1, 'efg_pct') - _get_eff(team2, 'efg_pct'),
        "to_rate_diff": _get_eff(team1, 'to_rate') - _get_eff(team2, 'to_rate'),
        "orb_diff": _get_eff(team1, 'orb_pct') - _get_eff(team2, 'orb_pct'),
        "ft_rate_diff": _get_eff(team1, 'ft_rate') - _get_eff(team2, 'ft_rate'),
    }


def calculate_total_features(team1: dict, team2: dict) -> dict:
    """Features for predicting game total."""
    tempo1 = _get_eff(team1, 'tempo')
    tempo2 = _get_eff(team2, 'tempo')

    return {
        "combined_tempo": tempo1 + tempo2,
        "avg_tempo": (tempo1 + tempo2) / 2,
        "combined_ppg": _get_eff(team1, 'ppg') + _get_eff(team2, 'ppg'),
        "combined_opp_ppg": _get_eff(team1, 'opp_ppg') + _get_eff(team2, 'opp_ppg'),
        "combined_off_eff": _get_eff(team1, 'off_rating') + _get_eff(team2, 'off_rating'),
        "combined_def_eff": _get_eff(team1, 'def_rating') + _get_eff(team2, 'def_rating'),
        "combined_fg_pct": _get_eff(team1, 'fg_pct') + _get_eff(team2, 'fg_pct'),
        "combined_3pt_pct": _get_eff(team1, 'three_pct') + _get_eff(team2, 'three_pct'),
        "projected_total": project_game_total(team1, team2),
    }


def calculate_win_probability_features(team1: dict, team2: dict) -> dict:
    """Features for win probability prediction."""
    return {
        "net_rating_diff": _get_eff(team1, 'net_rating') - _get_eff(team2, 'net_rating'),
        "seed_diff": _get_eff(team1, 'seed') - _get_eff(team2, 'seed'),
        "ranking_diff": _get_eff(team1, 'ranking') - _get_eff(team2, 'ranking'),
        "tournament_exp_diff": _get_eff(team1, 'tourney_appearances') - _get_eff(team2, 'tourney_appearances'),
        "last_10_win_pct_diff": _get_eff(team1, 'last_10_pct') - _get_eff(team2, 'last_10_pct'),
        "sos_adjusted_margin": _get_eff(team1, 'margin') * _get_eff(team1, 'sos') - _get_eff(team2, 'margin') * _get_eff(team2, 'sos'),
    }


def calculate_implied_probability(moneyline: int) -> float:
    """Convert American odds to implied probability."""
    try:
        ml = int(moneyline)
    except Exception:
        return 0.5

    if ml > 0:
        return 100.0 / (ml + 100.0)
    else:
        return abs(ml) / (abs(ml) + 100.0)


def find_value_bets(predictions: List[dict], lines: List[dict], threshold: float = 0.05) -> List[dict]:
    """Find games where model probability exceeds implied probability."""
    value_bets = []
    for pred, line in zip(predictions, lines):
        model_prob = pred.get('win_prob') if pred.get('win_prob') is not None else pred.get('model_prob')
        if model_prob is None:
            continue
        ml = line.get('moneyline') or line.get('ml') or 0
        implied_prob = calculate_implied_probability(ml)
        edge = float(model_prob) - float(implied_prob)
        if edge > threshold:
            value_bets.append({
                "team": pred.get('team') or pred.get('home') or pred.get('away'),
                "model_prob": float(model_prob),
                "implied_prob": float(implied_prob),
                "edge": float(edge),
                "moneyline": ml,
            })

    return sorted(value_bets, key=lambda x: -x['edge'])


def calculate_ats_features(team1: dict, team2: dict, spread: float) -> dict:
    """Features for predicting ATS outcomes."""
    predicted_margin = _get_eff(team1, 'predicted_margin')
    return {
        "predicted_margin": predicted_margin,
        "spread": float(spread),
        "spread_diff": predicted_margin - float(spread),
        "team1_ats_pct": _get_eff(team1, 'ats_pct', default=0.5),
        "team2_ats_pct": _get_eff(team2, 'ats_pct', default=0.5),
        "team1_avg_cover_margin": _get_eff(team1, 'avg_cover', default=0.0),
        "team2_avg_cover_margin": _get_eff(team2, 'avg_cover', default=0.0),
        "fav_ats_pct": _get_eff(team1, 'favorite_ats_pct', default=0.5),
        "dog_ats_pct": _get_eff(team2, 'underdog_ats_pct', default=0.5),
    }


__all__ = [
    'calculate_efficiency_differential',
    'project_game_total',
    'calculate_spread_features',
    'calculate_total_features',
    'calculate_win_probability_features',
    'calculate_implied_probability',
    'find_value_bets',
    'calculate_ats_features',
]
