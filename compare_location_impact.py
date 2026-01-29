#!/usr/bin/env python3
"""Compare model performance with and without location features."""

import numpy as np
from upset_prediction import UpsetPredictor, HISTORICAL_UPSET_RATES
from sklearn.model_selection import train_test_split

def create_training_data_without_location(n_samples=1000):
    """Create training data WITHOUT location features."""
    np.random.seed(42)
    
    features = []
    targets = []
    
    for _ in range(n_samples):
        fav_seed = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        und_seed = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16])
        
        favorite = {
            'seed': fav_seed,
            'net_efficiency': np.random.normal(15, 5),
            'tempo': np.random.normal(70, 5),
            'three_rate': np.random.normal(0.35, 0.05),
            'def_efficiency': np.random.normal(100, 5),
            'coach_ncaa_games': np.random.poisson(50),
            'last_10_wins': np.random.poisson(6),
            'conf_strength': np.random.choice([1, 2, 3, 4, 5])
        }
        
        underdog = {
            'seed': und_seed,
            'net_efficiency': np.random.normal(10, 5),
            'tempo': np.random.normal(68, 5),
            'three_rate': np.random.normal(0.38, 0.05),
            'def_efficiency': np.random.normal(98, 5),
            'coach_ncaa_games': np.random.poisson(30),
            'last_10_wins': np.random.poisson(7),
            'conf_strength': np.random.choice([1, 2, 3, 4, 5])
        }
        
        game_context = {'round_number': np.random.choice([1, 2, 3, 4])}
        
        predictor = UpsetPredictor()
        feature_vector = predictor.create_features(favorite, underdog, game_context)
        
        seed_pair = (fav_seed, und_seed)
        base_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.3)
        eff_diff = underdog['net_efficiency'] - favorite['net_efficiency']
        adjusted_rate = base_rate + (eff_diff * 0.01)
        
        upset_occurred = np.random.random() < min(max(adjusted_rate, 0.01), 0.99)
        
        features.append(feature_vector)
        targets.append(1 if upset_occurred else 0)
    
    return np.array(features), np.array(targets)

def create_training_data_with_location(n_samples=1000):
    """Create training data WITH location features."""
    np.random.seed(42)
    
    features = []
    targets = []
    
    for _ in range(n_samples):
        from math import sqrt
        
        def distance(loc1, loc2):
            return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
        
        fav_seed = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        und_seed = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16])
        
        game_location = (np.random.uniform(25, 48), np.random.uniform(-120, -70))
        fav_school_location = (np.random.uniform(25, 48), np.random.uniform(-120, -70))
        und_school_location = (np.random.uniform(25, 48), np.random.uniform(-120, -70))
        
        favorite = {
            'seed': fav_seed,
            'net_efficiency': np.random.normal(15, 5),
            'tempo': np.random.normal(70, 5),
            'three_rate': np.random.normal(0.35, 0.05),
            'def_efficiency': np.random.normal(100, 5),
            'coach_ncaa_games': np.random.poisson(50),
            'last_10_wins': np.random.poisson(6),
            'conf_strength': np.random.choice([1, 2, 3, 4, 5]),
            'school_location': fav_school_location
        }
        
        underdog = {
            'seed': und_seed,
            'net_efficiency': np.random.normal(10, 5),
            'tempo': np.random.normal(68, 5),
            'three_rate': np.random.normal(0.38, 0.05),
            'def_efficiency': np.random.normal(98, 5),
            'coach_ncaa_games': np.random.poisson(30),
            'last_10_wins': np.random.poisson(7),
            'conf_strength': np.random.choice([1, 2, 3, 4, 5]),
            'school_location': und_school_location
        }
        
        game_context = {
            'round_number': np.random.choice([1, 2, 3, 4]),
            'location': game_location
        }
        
        predictor = UpsetPredictor()
        feature_vector = predictor.create_features(favorite, underdog, game_context)
        
        seed_pair = (fav_seed, und_seed)
        base_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.3)
        eff_diff = underdog['net_efficiency'] - favorite['net_efficiency']
        adjusted_rate = base_rate + (eff_diff * 0.01)
        
        # Add location effect
        und_dist = distance(und_school_location, game_location)
        fav_dist = distance(fav_school_location, game_location)
        
        if und_dist < fav_dist * 0.5:
            adjusted_rate += 0.03
        if und_dist < 5 and fav_dist > 20:
            adjusted_rate += 0.05
        
        upset_occurred = np.random.random() < min(max(adjusted_rate, 0.01), 0.99)
        
        features.append(feature_vector)
        targets.append(1 if upset_occurred else 0)
    
    return np.array(features), np.array(targets)

print("=" * 70)
print("COMPARING UPSET PREDICTION MODELS")
print("=" * 70)

# Train WITHOUT location features
print("\n1. Model WITHOUT Location Features:")
print("-" * 70)
X_no_loc, y_no_loc = create_training_data_without_location(1000)
print(f"Features shape: {X_no_loc.shape}")
print(f"Upset rate: {y_no_loc.mean():.1%}")

predictor_no_loc = UpsetPredictor()
metrics_no_loc = predictor_no_loc.train(X_no_loc, y_no_loc)

print(f"\nPerformance:")
print(f"  Test Accuracy: {metrics_no_loc['test_accuracy']:.4f}")
print(f"  Precision: {metrics_no_loc['precision']:.4f}")
print(f"  Recall: {metrics_no_loc['recall']:.4f}")
print(f"  F1 Score: {metrics_no_loc['f1_score']:.4f}")

# Train WITH location features
print("\n2. Model WITH Location Features:")
print("-" * 70)
X_with_loc, y_with_loc = create_training_data_with_location(1000)
print(f"Features shape: {X_with_loc.shape}")
print(f"Upset rate: {y_with_loc.mean():.1%}")

predictor_with_loc = UpsetPredictor()
metrics_with_loc = predictor_with_loc.train(X_with_loc, y_with_loc)

print(f"\nPerformance:")
print(f"  Test Accuracy: {metrics_with_loc['test_accuracy']:.4f}")
print(f"  Precision: {metrics_with_loc['precision']:.4f}")
print(f"  Recall: {metrics_with_loc['recall']:.4f}")
print(f"  F1 Score: {metrics_with_loc['f1_score']:.4f}")

# Comparison
print("\n3. Performance Comparison:")
print("=" * 70)
print(f"{'Metric':<20} {'Without Location':<20} {'With Location':<20} {'Change':<15}")
print("-" * 70)

metrics_to_compare = ['test_accuracy', 'precision', 'recall', 'f1_score']
for metric in metrics_to_compare:
    without = metrics_no_loc[metric]
    with_ = metrics_with_loc[metric]
    change = ((with_ - without) / without * 100) if without > 0 else 0
    print(f"{metric:<20} {without:<20.4f} {with_:<20.4f} {change:+.2f}%")

# Feature importance comparison
print("\n4. New Location Features Importance:")
print("-" * 70)
location_features = ['distance_diff', 'underdog_travel_advantage', 'underdog_pseudo_home']
for feat in location_features:
    if feat in metrics_with_loc['feature_importance']:
        importance = metrics_with_loc['feature_importance'][feat]
        print(f"  {feat}: {importance:.4f}")

print("\n5. Top 5 Features (WITH location):")
print("-" * 70)
importances = sorted(metrics_with_loc['feature_importance'].items(), 
                    key=lambda x: x[1], reverse=True)
for feature, importance in importances[:5]:
    print(f"  {feature}: {importance:.4f}")
