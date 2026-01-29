#!/usr/bin/env python3
"""Demonstrate the impact of location features with realistic examples."""

from upset_prediction import UpsetPredictor, train_upset_model
import joblib
from pathlib import Path

# Load the trained model
MODEL_DIR = Path('data_files/models')
try:
    predictor = UpsetPredictor()
    predictor.model = joblib.load(MODEL_DIR / 'upset_predictor.joblib')
    predictor.scaler = joblib.load(MODEL_DIR / 'upset_predictor_scaler.joblib')
    predictor.is_trained = True
    print("Loaded pre-trained model with location features\n")
except:
    print("Training new model...")
    predictor, _ = train_upset_model()

print("=" * 80)
print("LOCATION ADVANTAGE IMPACT EXAMPLES")
print("=" * 80)

# Example 1: Close matchup with neutral location
print("\nExample 1: (5) Michigan vs (12) Iowa State - Neutral Location (Denver)")
print("-" * 80)

michigan = {
    'seed': 5,
    'net_efficiency': 18.5,
    'tempo': 68.0,
    'three_rate': 0.36,
    'def_efficiency': 97.0,
    'coach_ncaa_games': 45,
    'last_10_wins': 7,
    'conf_strength': 5,
    'school_location': (42.28, -83.74)  # Ann Arbor, MI
}

iowa_state = {
    'seed': 12,
    'net_efficiency': 16.2,
    'tempo': 71.0,
    'three_rate': 0.42,
    'def_efficiency': 99.0,
    'coach_ncaa_games': 38,
    'last_10_wins': 8,
    'conf_strength': 4,
    'school_location': (42.03, -93.64)  # Ames, IA
}

# Neutral location
context_neutral = {
    'round_number': 1,
    'location': (39.74, -104.99)  # Denver, CO
}

result_neutral = predictor.predict_upset_probability(michigan, iowa_state, context_neutral)
print(f"Upset Probability: {result_neutral['upset_probability']:.1%}")
print(f"Confidence: {result_neutral['confidence']}")

# Example 2: Same matchup, but close to Iowa State
print("\nExample 2: Same teams - Game in Kansas City (near Iowa State)")
print("-" * 80)

context_near_iowa = {
    'round_number': 1,
    'location': (39.10, -94.58)  # Kansas City, MO (closer to Iowa State)
}

result_near_iowa = predictor.predict_upset_probability(michigan, iowa_state, context_near_iowa)
print(f"Upset Probability: {result_near_iowa['upset_probability']:.1%}")
print(f"Confidence: {result_near_iowa['confidence']}")
print(f"Change from neutral: {(result_near_iowa['upset_probability'] - result_neutral['upset_probability'])*100:+.1f} percentage points")

# Check location factors
location_analysis = predictor._analyze_location_advantage(
    michigan, iowa_state, context_near_iowa['location']
)
if location_analysis['travel_advantage']:
    print("✓ Iowa State has significant travel advantage")
if location_analysis['pseudo_home_game']:
    print("✓ Pseudo home game for Iowa State")

# Example 3: Same matchup, but close to Michigan
print("\nExample 3: Same teams - Game in Indianapolis (near Michigan)")
print("-" * 80)

context_near_michigan = {
    'round_number': 1,
    'location': (39.77, -86.16)  # Indianapolis, IN (closer to Michigan)
}

result_near_michigan = predictor.predict_upset_probability(michigan, iowa_state, context_near_michigan)
print(f"Upset Probability: {result_near_michigan['upset_probability']:.1%}")
print(f"Confidence: {result_near_michigan['confidence']}")
print(f"Change from neutral: {(result_near_michigan['upset_probability'] - result_neutral['upset_probability'])*100:+.1f} percentage points")

# Example 4: Extreme case - very close to underdog
print("\nExample 4: (6) Texas vs (11) VCU - Game in Richmond, VA (VCU's home)")
print("-" * 80)

texas = {
    'seed': 6,
    'net_efficiency': 17.8,
    'tempo': 69.0,
    'three_rate': 0.34,
    'def_efficiency': 98.0,
    'coach_ncaa_games': 52,
    'last_10_wins': 6,
    'conf_strength': 5,
    'school_location': (30.28, -97.74)  # Austin, TX
}

vcu = {
    'seed': 11,
    'net_efficiency': 15.5,
    'tempo': 73.0,
    'three_rate': 0.39,
    'def_efficiency': 100.0,
    'coach_ncaa_games': 35,
    'last_10_wins': 9,
    'conf_strength': 3,
    'school_location': (37.55, -77.45)  # Richmond, VA
}

context_vcu_home = {
    'round_number': 1,
    'location': (37.55, -77.45)  # Richmond, VA - VCU's actual location
}

result_vcu_home = predictor.predict_upset_probability(texas, vcu, context_vcu_home)
print(f"Upset Probability: {result_vcu_home['upset_probability']:.1%}")
print(f"Confidence: {result_vcu_home['confidence']}")

location_analysis_vcu = predictor._analyze_location_advantage(
    texas, vcu, context_vcu_home['location']
)
print(f"\nLocation Analysis:")
print(f"  Travel advantage for VCU: {location_analysis_vcu['travel_advantage']}")
print(f"  Pseudo home game for VCU: {location_analysis_vcu['pseudo_home_game']}")
print(f"  Upset probability boost: +{location_analysis_vcu['upset_probability_boost']:.1%}")

print("\n" + "=" * 80)
print("SUMMARY: How Location Features Affect the Model")
print("=" * 80)
print("""
The location features add 3 new inputs to the model:

1. distance_diff (continuous): Distance differential between teams to game site
   - Importance: 11.5% (5th most important feature)
   - Effect: Directly measures travel advantage
   
2. underdog_travel_advantage (binary): Underdog significantly closer (< 50% distance)
   - Importance: 0.4% 
   - Effect: Small boost when triggered
   
3. underdog_pseudo_home (binary): Underdog very close (< 100 mi) AND favorite far (> 500 mi)
   - Importance: ~0%
   - Effect: Rare but meaningful when it occurs (home crowd advantage)

Key Insights:
- Location CAN matter, especially when one team has a clear proximity advantage
- The continuous distance_diff feature is more useful than binary indicators
- In practice, NCAA tries to avoid giving teams home-court advantage in early rounds
- Later rounds (Elite 8, Final Four) are truly neutral sites with less location impact
- The model learns that location is less important than efficiency, tempo, and style
""")
