#!/usr/bin/env python3
"""
Test script for upset prediction model.

Demonstrates how to use the UpsetPredictor class with sample data.
"""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from upset_prediction import UpsetPredictor, train_upset_model


def test_upset_prediction():
    """Test the upset prediction functionality."""

    print("=" * 60)
    print("UPSET PREDICTION MODEL TEST")
    print("=" * 60)

    # Train the model
    print("\nTraining model on synthetic historical data...")
    model, results = train_upset_model()

    print("\n" + "=" * 60)
    print("TEST PREDICTIONS")
    print("=" * 60)

    # Test cases with different scenarios
    test_cases = [
        {
            'name': 'Classic David vs Goliath',
            'favorite': {
                'seed': 1,
                'net_efficiency': 25.0,
                'tempo': 72.0,
                'three_rate': 0.32,
                'def_efficiency': 95.0,
                'coach_ncaa_games': 80,
                'last_10_wins': 8,
                'conf_strength': 5
            },
            'underdog': {
                'seed': 16,
                'net_efficiency': 5.0,  # Much worse efficiency
                'tempo': 65.0,
                'three_rate': 0.42,
                'def_efficiency': 110.0,  # Much worse defense
                'coach_ncaa_games': 10,
                'last_10_wins': 3,
                'conf_strength': 1
            }
        },
        {
            'name': 'Realistic Upset Candidate',
            'favorite': {
                'seed': 4,
                'net_efficiency': 18.0,
                'tempo': 70.0,
                'three_rate': 0.35,
                'def_efficiency': 98.0,
                'coach_ncaa_games': 60,
                'last_10_wins': 6,
                'conf_strength': 4
            },
            'underdog': {
                'seed': 13,
                'net_efficiency': 20.0,  # Better efficiency than favorite!
                'tempo': 75.0,  # Much faster tempo
                'three_rate': 0.45,  # Heavy three-point shooting
                'def_efficiency': 96.0,  # Better defense
                'coach_ncaa_games': 40,
                'last_10_wins': 9,  # Hot streak
                'conf_strength': 3
            }
        },
        {
            'name': 'Toss-Up Matchup',
            'favorite': {
                'seed': 7,
                'net_efficiency': 12.0,
                'tempo': 68.0,
                'three_rate': 0.38,
                'def_efficiency': 102.0,
                'coach_ncaa_games': 45,
                'last_10_wins': 5,
                'conf_strength': 3
            },
            'underdog': {
                'seed': 10,
                'net_efficiency': 14.0,
                'tempo': 66.0,
                'three_rate': 0.40,
                'def_efficiency': 100.0,
                'coach_ncaa_games': 50,  # More experienced coach
                'last_10_wins': 7,
                'conf_strength': 4,
                'conf_tourney_champion': True  # Won conference tournament
            }
        }
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. {test_case['name']}")
        print("-" * 40)

        prediction = model.predict_upset_probability(
            test_case['favorite'],
            test_case['underdog']
        )

        fav_seed = test_case['favorite']['seed']
        und_seed = test_case['underdog']['seed']

        print(f"Matchup: ({und_seed}) Underdog vs ({fav_seed}) Favorite")
        print(f"Upset Probability: {prediction['upset_probability']:.1%}")
        print(f"Confidence: {prediction['confidence']}")
        print(f"Historical Rate: {prediction['historical_rate']:.1%}")

        if prediction['key_reasons']:
            print("Key Factors:")
            for reason in prediction['key_reasons'][:3]:
                print(f"  • {reason}")
        else:
            print("No significant upset factors identified")

        # Show efficiency comparison
        fav_eff = test_case['favorite']['net_efficiency']
        und_eff = test_case['underdog']['net_efficiency']
        print(f"Efficiency: Underdog {und_eff:.1f} vs Favorite {fav_eff:.1f} "
              f"(Diff: {und_eff - fav_eff:+.1f})")


def test_feature_importance():
    """Test and display feature importance analysis."""

    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)

    model, results = train_upset_model()

    print("\nAll Features (ranked by importance):")
    sorted_features = sorted(results['feature_importance'].items(),
                           key=lambda x: x[1], reverse=True)

    for i, (feature, importance) in enumerate(sorted_features, 1):
        print(f"{i:2d}. {feature}: {importance:.3f}")


if __name__ == "__main__":
    test_upset_prediction()
    test_feature_importance()