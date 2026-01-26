#!/usr/bin/env python3
"""
Upset Prediction Model

Specialized model for predicting tournament upsets in March Madness.
Trains on historical tournament data to identify games with high upset potential.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# Historical upset rates by seed matchup (from NCAA tournament data)
HISTORICAL_UPSET_RATES = {
    (1, 16): 0.01,   # Almost never happens (UMBC was first)
    (2, 15): 0.06,   # Rare but happens
    (3, 14): 0.15,   # Occasional
    (4, 13): 0.21,   # Common upset spot
    (5, 12): 0.35,   # Famous "5-12 upset" territory
    (6, 11): 0.37,   # Very common
    (7, 10): 0.39,   # Basically a toss-up
    (8, 9): 0.49,    # True toss-up
}


class UpsetPredictor:
    """Specialized model for predicting tournament upsets."""

    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=20,
            subsample=0.8,
            random_state=42
        )
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False

    def create_features(self, favorite: dict, underdog: dict,
                        game_context: dict = None) -> np.array:
        """Create feature vector for upset prediction.

        Args:
            favorite: Dict with higher seed team data
            underdog: Dict with lower seed team data
            game_context: Optional game context (round, location, etc.)

        Returns:
            Feature vector as numpy array
        """

        features = []
        self.feature_names = []

        # Seed differential (smaller = more likely upset)
        seed_diff = underdog.get('seed', 8) - favorite.get('seed', 1)
        features.append(seed_diff)
        self.feature_names.append('seed_diff')

        # Efficiency metrics
        net_eff_diff = (underdog.get('net_efficiency', 0) -
                       favorite.get('net_efficiency', 0))
        features.append(net_eff_diff)
        self.feature_names.append('net_eff_diff')

        # Binary indicator for underdog better efficiency
        features.append(1 if net_eff_diff > 0 else 0)
        self.feature_names.append('underdog_better_eff')

        # Style factors
        tempo_diff = abs(favorite.get('tempo', 70) - underdog.get('tempo', 70))
        features.append(tempo_diff)
        self.feature_names.append('tempo_diff')

        # Three-point shooting
        features.append(underdog.get('three_rate', 0.35))
        self.feature_names.append('underdog_three_rate')

        # Defensive efficiency (lower is better)
        features.append(underdog.get('def_efficiency', 100))
        self.feature_names.append('underdog_def_eff')

        # Experience factors
        coach_exp_diff = (underdog.get('coach_ncaa_games', 0) -
                         favorite.get('coach_ncaa_games', 0))
        features.append(coach_exp_diff)
        self.feature_names.append('coach_exp_diff')

        # Momentum (last 10 games)
        features.append(underdog.get('last_10_wins', 5))
        self.feature_names.append('underdog_last_10')

        features.append(favorite.get('last_10_wins', 5))
        self.feature_names.append('favorite_last_10')

        # Conference strength (proxy for schedule strength)
        features.append(underdog.get('conf_strength', 5))
        self.feature_names.append('underdog_conf_strength')

        # Round number (upsets less common in later rounds)
        round_num = game_context.get('round_number', 1) if game_context else 1
        features.append(round_num)
        self.feature_names.append('round_number')

        # Location advantage (if available)
        if game_context and 'location_advantage' in game_context:
            features.append(game_context['location_advantage'])
            self.feature_names.append('location_advantage')

        return np.array(features)

    def train(self, X: np.array, y: np.array) -> Dict[str, Any]:
        """Train the upset prediction model.

        Args:
            X: Feature matrix
            y: Binary target (1 for upset, 0 for no upset)

        Returns:
            Training results dictionary
        """

        # Set feature names by calling create_features once
        # This ensures feature_names is populated on this instance
        dummy_fav = {'seed': 1}
        dummy_und = {'seed': 16}
        _ = self.create_features(dummy_fav, dummy_und)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=42
        )

        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)

        # Predictions for detailed metrics
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            )),
            'feature_names': self.feature_names
        }

    def predict_upset_probability(self, favorite: dict, underdog: dict,
                                  game_context: dict = None) -> Dict[str, Any]:
        """Predict probability of an upset.

        Args:
            favorite: Dict with higher seed team data
            underdog: Dict with lower seed team data
            game_context: Optional game context

        Returns:
            Dictionary with prediction results
        """

        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Create features
        features = self.create_features(favorite, underdog, game_context)
        features_scaled = self.scaler.transform(features.reshape(1, -1))

        # Get prediction
        proba = self.model.predict_proba(features_scaled)[0]
        upset_prob = proba[1]  # Probability of upset

        # Analyze contributing factors
        efficiency = self._check_efficiency_mismatch(favorite, underdog)
        style = self._analyze_style_matchup(favorite, underdog)
        experience = self._analyze_experience_momentum(favorite, underdog)

        # Get historical rate
        seed_pair = (favorite.get('seed', 1), underdog.get('seed', 16))
        historical_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.3)

        return {
            'upset_probability': upset_prob,
            'confidence': 'High' if upset_prob > 0.4 else 'Medium' if upset_prob > 0.25 else 'Low',
            'factors': {
                'efficiency_mismatch': efficiency,
                'style_matchup': style,
                'experience_momentum': experience
            },
            'key_reasons': self._get_key_reasons(efficiency, style, experience),
            'historical_rate': historical_rate,
            'seed_pair': seed_pair
        }

    def _check_efficiency_mismatch(self, favorite: dict, underdog: dict) -> Dict[str, Any]:
        """Check if lower seed is actually better by efficiency metrics."""

        result = {
            'net_eff_favors_underdog': False,
            'off_eff_favors_underdog': False,
            'def_eff_favors_underdog': False,
            'upset_indicator_strength': 0.0
        }

        # Net efficiency comparison
        net_diff = underdog.get('net_efficiency', 0) - favorite.get('net_efficiency', 0)
        if net_diff > 0:
            result['net_eff_favors_underdog'] = True
            result['upset_indicator_strength'] += 0.4

        # Offensive efficiency
        off_diff = underdog.get('off_efficiency', 0) - favorite.get('off_efficiency', 0)
        if off_diff > 2:
            result['off_eff_favors_underdog'] = True
            result['upset_indicator_strength'] += 0.2

        # Defensive efficiency (lower is better)
        def_diff = favorite.get('def_efficiency', 100) - underdog.get('def_efficiency', 100)
        if def_diff > 2:
            result['def_eff_favors_underdog'] = True
            result['upset_indicator_strength'] += 0.2

        return result

    def _analyze_style_matchup(self, favorite: dict, underdog: dict) -> Dict[str, Any]:
        """Analyze if the underdog's style matches up well against the favorite."""

        factors = {
            'tempo_advantage': False,
            'three_point_variance': False,
            'defensive_disruption': False,
            'upset_probability_boost': 0.0
        }

        # Tempo mismatch - if underdog wants to slow down a fast team
        # or speed up a slow team, they can disrupt
        tempo_diff = abs(favorite.get('tempo', 70) - underdog.get('tempo', 70))
        if tempo_diff > 5:
            factors['tempo_advantage'] = True
            factors['upset_probability_boost'] += 0.05

        # Three-point shooting variance
        # Teams that live by the three can upset (or lose badly)
        if underdog.get('three_rate', 0.35) > 0.40:
            factors['three_point_variance'] = True
            factors['upset_probability_boost'] += 0.03

        # Strong defensive team can limit favorite's efficiency
        if underdog.get('def_efficiency', 100) < 98:
            factors['defensive_disruption'] = True
            factors['upset_probability_boost'] += 0.05

        return factors

    def _analyze_experience_momentum(self, favorite: dict, underdog: dict) -> Dict[str, Any]:
        """Experience and recent form matter in tournament."""

        factors = {
            'coach_advantage': False,
            'conference_tourney_winner': False,
            'hot_streak': False,
            'favorite_cold': False,
            'upset_probability_boost': 0.0
        }

        # Coach tournament experience
        fav_coach_exp = favorite.get('coach_ncaa_games', 0)
        und_coach_exp = underdog.get('coach_ncaa_games', 0)

        if und_coach_exp > fav_coach_exp:
            factors['coach_advantage'] = True
            factors['upset_probability_boost'] += 0.03

        # Conference tournament momentum
        if underdog.get('conf_tourney_champion', False):
            factors['conference_tourney_winner'] = True
            factors['upset_probability_boost'] += 0.05

        # Recent form (last 10 games)
        und_last_10 = underdog.get('last_10_wins', 5)
        fav_last_10 = favorite.get('last_10_wins', 5)

        if und_last_10 >= 8:
            factors['hot_streak'] = True
            factors['upset_probability_boost'] += 0.03

        if fav_last_10 <= 5:
            factors['favorite_cold'] = True
            factors['upset_probability_boost'] += 0.05

        return factors

    def _get_key_reasons(self, efficiency: dict, style: dict,
                         experience: dict) -> List[str]:
        """Get human-readable reasons for upset potential."""

        reasons = []

        if efficiency['net_eff_favors_underdog']:
            reasons.append("Underdog has better efficiency rating")

        if style['tempo_advantage']:
            reasons.append("Style mismatch favors underdog")

        if style['three_point_variance']:
            reasons.append("Underdog's three-point shooting adds variance")

        if experience['coach_advantage']:
            reasons.append("Underdog coach has more tournament experience")

        if experience['hot_streak']:
            reasons.append("Underdog is on a hot streak")

        if experience['favorite_cold']:
            reasons.append("Favorite has been struggling recently")

        if experience['conference_tourney_winner']:
            reasons.append("Underdog won their conference tournament")

        return reasons


def create_historical_training_data() -> Tuple[np.array, np.array]:
    """Create training data from historical tournament results.

    This is a placeholder - in practice, you'd load real historical data.
    For now, we'll create synthetic data based on historical patterns.
    """

    # Generate synthetic training data based on historical patterns
    n_samples = 1000
    features = []
    targets = []

    for _ in range(n_samples):
        # Random seed matchup
        fav_seed = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        und_seed = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16])

        # Create team stats (simplified)
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

        # Create features
        predictor = UpsetPredictor()
        feature_vector = predictor.create_features(favorite, underdog)

        # Determine if upset occurred (based on historical rates + some noise)
        seed_pair = (fav_seed, und_seed)
        base_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.3)

        # Adjust based on efficiency difference
        eff_diff = underdog['net_efficiency'] - favorite['net_efficiency']
        adjusted_rate = base_rate + (eff_diff * 0.01)  # 1% per efficiency point

        # Add some random noise
        upset_occurred = np.random.random() < min(max(adjusted_rate, 0.01), 0.99)

        features.append(feature_vector)
        targets.append(1 if upset_occurred else 0)

    return np.array(features), np.array(targets)


def train_upset_model() -> Tuple[UpsetPredictor, Dict[str, Any]]:
    """Train the upset prediction model and return results."""

    print("Creating historical training data...")
    X, y = create_historical_training_data()

    print(f"Training data shape: {X.shape}")
    print(f"Upset rate in training data: {y.mean():.1%}")

    print("\nTraining upset prediction model...")
    predictor = UpsetPredictor()
    results = predictor.train(X, y)

    print("\nTraining Results:")
    print(f"  Train Accuracy: {results['train_accuracy']:.3f}")
    print(f"  Test Accuracy: {results['test_accuracy']:.3f}")
    print(f"  Precision: {results['precision']:.3f}")
    print(f"  Recall: {results['recall']:.3f}")
    print(f"  F1 Score: {results['f1_score']:.3f}")

    print("\nTop 5 Most Important Features:")
    sorted_features = sorted(results['feature_importance'].items(),
                           key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_features[:5]:
        print(f"  {feature}: {importance:.3f}")

    return predictor, results


if __name__ == "__main__":
    # Train the model
    model, results = train_upset_model()

    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)

    # Example teams (simplified data)
    favorite = {
        'seed': 1,
        'net_efficiency': 25.0,
        'tempo': 72.0,
        'three_rate': 0.32,
        'def_efficiency': 95.0,
        'coach_ncaa_games': 80,
        'last_10_wins': 8,
        'conf_strength': 5
    }

    underdog = {
        'seed': 16,
        'net_efficiency': 18.0,
        'tempo': 65.0,
        'three_rate': 0.42,
        'def_efficiency': 102.0,
        'coach_ncaa_games': 20,
        'last_10_wins': 9,
        'conf_strength': 2
    }

    prediction = model.predict_upset_probability(favorite, underdog)

    print(f"Matchup: ({underdog['seed']}) Underdog vs ({favorite['seed']}) Favorite")
    print(f"Upset Probability: {prediction['upset_probability']:.1%}")
    print(f"Confidence: {prediction['confidence']}")
    print(f"Historical Rate: {prediction['historical_rate']:.1%}")
    print(f"Key Reasons:")
    for reason in prediction['key_reasons'][:3]:
        print(f"  • {reason}")