"""Small sanity test for features.py functions."""
import sys
from pathlib import Path

# Ensure project root is on path when running from scripts/
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features import (
    calculate_efficiency_differential,
    project_game_total,
    calculate_spread_features,
    calculate_total_features,
    calculate_win_probability_features,
    calculate_implied_probability,
    find_value_bets,
    calculate_ats_features,
)


def main():
    t1 = {
        'adj_offense': 110.0,
        'adj_defense': 95.0,
        'tempo': 72,
        'ppg': 78,
        'opp_ppg': 68,
        'net_rating': 15,
        'seed': 3,
        'ranking': 12,
        'tourney_appearances': 5,
        'last_10_pct': 0.7,
        'margin': 8.5,
        'sos': 1.02,
    }

    t2 = {
        'adj_offense': 102.0,
        'adj_defense': 98.0,
        'tempo': 68,
        'ppg': 72,
        'opp_ppg': 69,
        'net_rating': 4,
        'seed': 7,
        'ranking': 45,
        'tourney_appearances': 2,
        'last_10_pct': 0.4,
        'margin': 1.2,
        'sos': 0.98,
    }

    print('eff diff', calculate_efficiency_differential(t1, t2))
    print('proj total', project_game_total(t1, t2))
    print('spread feats', calculate_spread_features(t1, t2))
    print('total feats', calculate_total_features(t1, t2))
    print('win feats', calculate_win_probability_features(t1, t2))
    print('implied + ML', calculate_implied_probability(150), calculate_implied_probability(-200))
    preds = [{'team': 'T1', 'win_prob': 0.7}, {'team': 'T2', 'win_prob': 0.3}]
    lines = [{'moneyline': 150}, {'moneyline': -120}]
    print('value bets', find_value_bets(preds, lines, threshold=0.05))
    print('ats', calculate_ats_features({'predicted_margin': 5}, {'predicted_margin': -2}, 3))


if __name__ == '__main__':
    main()
