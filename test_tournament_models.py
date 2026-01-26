"""Simple tests for tournament_models module.

These are lightweight checks to ensure the feature functions and adjustments
behave as expected without requiring heavy ML package training.
"""
import pytest
import numpy as np
from tournament_models import (
    create_tournament_features,
    calculate_upset_features,
    adjust_for_round,
    apply_seed_prior,
)


def test_create_tournament_features_shape():
    team1 = {'net_efficiency': 10, 'off_efficiency': 110, 'def_efficiency': 100, 'seed': 3}
    team2 = {'net_efficiency': 5, 'off_efficiency': 105, 'def_efficiency': 102, 'seed': 14}
    ctx = {'round_number': 2, 'travel_advantage': 0}

    feats = create_tournament_features(team1, team2, ctx)
    assert isinstance(feats, np.ndarray)
    assert feats.shape[0] == 14


def test_calculate_upset_features_length():
    fav = {'seed': 3, 'net_efficiency': 8, 'tempo': 70, 'three_rate': 0.35, 'def_efficiency': 95}
    under = {'seed': 14, 'net_efficiency': 10, 'tempo': 72, 'three_rate': 0.36, 'def_efficiency': 98}

    up_feats = calculate_upset_features(fav, under)
    assert isinstance(up_feats, np.ndarray)
    assert up_feats.shape[0] >= 7


def test_adjust_for_round_bounds():
    p = 0.6
    adjusted = adjust_for_round(p, round_number=5, team1_seed=1, team2_seed=8)
    assert 0.6 <= adjusted <= 0.95


def test_apply_seed_prior_blending():
    model_prob = 0.6
    blended = apply_seed_prior(model_prob, seed1=1, seed2=8, prior_weight=0.2)
    assert blended != model_prob

if __name__ == '__main__':
    pytest.main(['-q'])
