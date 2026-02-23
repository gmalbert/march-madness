#!/usr/bin/env python3
"""
Upset Prediction Model

Specialized model for predicting tournament upsets in March Madness.
Trains on historical tournament data to identify games with high upset potential.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# Utility functions for upset detection
def is_upset(winner_seed: int, loser_seed: int) -> bool:
    """An upset occurs when the higher-seeded team loses."""
    return winner_seed > loser_seed


def upset_magnitude(winner_seed: int, loser_seed: int) -> int:
    """How big is the upset? Larger = more surprising."""
    return winner_seed - loser_seed if is_upset(winner_seed, loser_seed) else 0


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

        def safe_get(d, key, default=0):
            """Safely get value from dict, handling None values."""
            val = d.get(key, default)
            return val if val is not None else default

        features = []
        self.feature_names = []

        # Seed differential (smaller = more likely upset)
        seed_diff = safe_get(underdog, 'seed', 8) - safe_get(favorite, 'seed', 1)
        features.append(seed_diff)
        self.feature_names.append('seed_diff')

        # Efficiency metrics
        net_eff_diff = (safe_get(underdog, 'net_efficiency', 0) -
                       safe_get(favorite, 'net_efficiency', 0))
        features.append(net_eff_diff)
        self.feature_names.append('net_eff_diff')

        # Binary indicator for underdog better efficiency
        features.append(1 if net_eff_diff > 0 else 0)
        self.feature_names.append('underdog_better_eff')

        # Style factors
        tempo_diff = abs(safe_get(favorite, 'tempo', 70) - safe_get(underdog, 'tempo', 70))
        features.append(tempo_diff)
        self.feature_names.append('tempo_diff')

        # Three-point shooting
        features.append(safe_get(underdog, 'three_rate', 0.35))
        self.feature_names.append('underdog_three_rate')

        # Defensive efficiency (lower is better)
        features.append(safe_get(underdog, 'def_efficiency', 100))
        self.feature_names.append('underdog_def_eff')

        # Experience factors
        coach_exp_diff = (safe_get(underdog, 'coach_ncaa_games', 0) -
                         safe_get(favorite, 'coach_ncaa_games', 0))
        features.append(coach_exp_diff)
        self.feature_names.append('coach_exp_diff')

        # Momentum (last 10 games)
        features.append(safe_get(underdog, 'last_10_wins', 5))
        self.feature_names.append('underdog_last_10')

        features.append(safe_get(favorite, 'last_10_wins', 5))
        self.feature_names.append('favorite_last_10')

        # Conference strength (proxy for schedule strength)
        features.append(safe_get(underdog, 'conf_strength', 5))
        self.feature_names.append('underdog_conf_strength')

        # Round number (upsets less common in later rounds)
        round_num = game_context.get('round_number', 1) if game_context else 1
        features.append(round_num)
        self.feature_names.append('round_number')

        # Location/Travel features
        # Calculate distance-based features if location data available
        if game_context and 'location' in game_context:
            from math import sqrt
            
            def distance(loc1: tuple, loc2: tuple) -> float:
                return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
            
            game_loc = game_context['location']
            fav_loc = favorite.get('school_location', (0, 0))
            und_loc = underdog.get('school_location', (0, 0))
            
            fav_dist = distance(fav_loc, game_loc)
            und_dist = distance(und_loc, game_loc)
            
            # Distance differential (negative = underdog closer)
            dist_diff = und_dist - fav_dist
            features.append(dist_diff)
            self.feature_names.append('distance_diff')
            
            # Binary: underdog has significant travel advantage
            features.append(1 if und_dist < fav_dist * 0.5 else 0)
            self.feature_names.append('underdog_travel_advantage')
            
            # Binary: pseudo home game for underdog
            features.append(1 if (und_dist < 100 and fav_dist > 500) else 0)
            self.feature_names.append('underdog_pseudo_home')
        else:
            # No location data - use neutral values
            features.append(0)
            self.feature_names.append('distance_diff')
            features.append(0)
            self.feature_names.append('underdog_travel_advantage')
            features.append(0)
            self.feature_names.append('underdog_pseudo_home')

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
        location = self._analyze_location_advantage(favorite, underdog,
                                                   game_context.get('location') if game_context else None)

        # Get historical rate
        seed_pair = (favorite.get('seed', 1), underdog.get('seed', 16))
        historical_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.3)

        return {
            'upset_probability': upset_prob,
            'confidence': 'High' if upset_prob > 0.4 else 'Medium' if upset_prob > 0.25 else 'Low',
            'factors': {
                'efficiency_mismatch': efficiency,
                'style_matchup': style,
                'experience_momentum': experience,
                'location_advantage': location
            },
            'key_reasons': self._get_key_reasons(efficiency, style, experience, location),
            'historical_rate': historical_rate,
            'seed_pair': seed_pair
        }

    def _check_efficiency_mismatch(self, favorite: dict, underdog: dict) -> Dict[str, Any]:
        """Check if lower seed is actually better by efficiency metrics."""

        def safe_get(d, key, default=0):
            """Safely get value from dict, handling None values."""
            val = d.get(key, default)
            return val if val is not None else default

        result = {
            'net_eff_favors_underdog': False,
            'off_eff_favors_underdog': False,
            'def_eff_favors_underdog': False,
            'upset_indicator_strength': 0.0
        }

        # Net efficiency comparison
        net_diff = safe_get(underdog, 'net_efficiency', 0) - safe_get(favorite, 'net_efficiency', 0)
        if net_diff > 0:
            result['net_eff_favors_underdog'] = True
            result['upset_indicator_strength'] += 0.4

        # Offensive efficiency
        off_diff = safe_get(underdog, 'off_efficiency', 0) - safe_get(favorite, 'off_efficiency', 0)
        if off_diff > 2:
            result['off_eff_favors_underdog'] = True
            result['upset_indicator_strength'] += 0.2

        # Defensive efficiency (lower is better)
        def_diff = safe_get(favorite, 'def_efficiency', 100) - safe_get(underdog, 'def_efficiency', 100)
        if def_diff > 2:
            result['def_eff_favors_underdog'] = True
            result['upset_indicator_strength'] += 0.2

        return result

    def _analyze_style_matchup(self, favorite: dict, underdog: dict) -> Dict[str, Any]:
        """Analyze if the underdog's style matches up well against the favorite."""

        def safe_get(d, key, default=0):
            """Safely get value from dict, handling None values."""
            val = d.get(key, default)
            return val if val is not None else default

        factors = {
            'tempo_advantage': False,
            'three_point_variance': False,
            'defensive_disruption': False,
            'upset_probability_boost': 0.0
        }

        # Tempo mismatch - if underdog wants to slow down a fast team
        # or speed up a slow team, they can disrupt
        tempo_diff = abs(safe_get(favorite, 'tempo', 70) - safe_get(underdog, 'tempo', 70))
        if tempo_diff > 5:
            factors['tempo_advantage'] = True
            factors['upset_probability_boost'] += 0.05

        # Three-point shooting variance
        # Teams that live by the three can upset (or lose badly)
        if safe_get(underdog, 'three_rate', 0.35) > 0.40:
            factors['three_point_variance'] = True
            factors['upset_probability_boost'] += 0.03

        # Strong defensive team can limit favorite's efficiency
        if safe_get(underdog, 'def_efficiency', 100) < 98:
            factors['defensive_disruption'] = True
            factors['upset_probability_boost'] += 0.05

        return factors

    def _analyze_experience_momentum(self, favorite: dict, underdog: dict) -> Dict[str, Any]:
        """Experience and recent form matter in tournament."""

        def safe_get(d, key, default=0):
            """Safely get value from dict, handling None values."""
            val = d.get(key, default)
            return val if val is not None else default

        factors = {
            'coach_advantage': False,
            'conference_tourney_winner': False,
            'hot_streak': False,
            'favorite_cold': False,
            'upset_probability_boost': 0.0
        }

        # Coach tournament experience
        fav_coach_exp = safe_get(favorite, 'coach_ncaa_games', 0)
        und_coach_exp = safe_get(underdog, 'coach_ncaa_games', 0)

        if und_coach_exp > fav_coach_exp:
            factors['coach_advantage'] = True
            factors['upset_probability_boost'] += 0.03

        # Conference tournament momentum
        if safe_get(underdog, 'conf_tourney_champion', False):
            factors['conference_tourney_winner'] = True
            factors['upset_probability_boost'] += 0.05

        # Recent form (last 10 games)
        und_last_10 = safe_get(underdog, 'last_10_wins', 5)
        fav_last_10 = safe_get(favorite, 'last_10_wins', 5)

        if und_last_10 >= 8:
            factors['hot_streak'] = True
            factors['upset_probability_boost'] += 0.03

        if fav_last_10 <= 5:
            factors['favorite_cold'] = True
            factors['upset_probability_boost'] += 0.05

        return factors

    def _analyze_location_advantage(self, favorite: dict, underdog: dict,
                                    game_location: tuple = None) -> Dict[str, Any]:
        """Analyze location and travel advantages."""
        from math import sqrt

        def distance(loc1: tuple, loc2: tuple) -> float:
            """Simple Euclidean distance (would use haversine for real)."""
            return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)

        factors = {
            'travel_advantage': False,
            'pseudo_home_game': False,
            'upset_probability_boost': 0.0
        }

        if not game_location:
            return factors

        fav_loc = favorite.get('school_location', (0, 0))
        und_loc = underdog.get('school_location', (0, 0))

        fav_distance = distance(fav_loc, game_location)
        und_distance = distance(und_loc, game_location)

        # Significant travel advantage
        if und_distance < fav_distance * 0.5:
            factors['travel_advantage'] = True
            factors['upset_probability_boost'] += 0.03

        # Pseudo home game (< 100 miles)
        if und_distance < 100 and fav_distance > 500:
            factors['pseudo_home_game'] = True
            factors['upset_probability_boost'] += 0.05

        return factors

    def _get_key_reasons(self, efficiency: dict, style: dict,
                         experience: dict, location: dict = None) -> List[str]:
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

        if location and location['travel_advantage']:
            reasons.append("Underdog has significant travel advantage")

        if location and location['pseudo_home_game']:
            reasons.append("Underdog is playing pseudo home game")

        return reasons


def create_training_data_from_csv(csv_path: str = None) -> Tuple[Optional[np.array], Optional[np.array]]:
    """Load real historical tournament data and convert to feature/target arrays.

    The CSV requires at least: actual_spread, home_score, away_score, net_eff_diff.
    Additional columns (kenpom_adjt_diff, def_eff_diff, etc.) improve features when present.

    Upset label: underdog wins — home team wins despite negative spread, or away wins
    despite positive spread.

    Returns:
        (X, y) numpy arrays aligned with UpsetPredictor.create_features() output,
        or (None, None) if data cannot be loaded.
    """
    # Candidate paths (relative to cwd)
    candidates = [
        csv_path,
        'data_files/training_data_complete_features.csv',
        'data_files/training_data_tournament.csv',
    ]
    found_path = None
    for p in candidates:
        if p and os.path.exists(p):
            found_path = p
            break
    if found_path is None:
        return None, None

    try:
        df = pd.read_csv(found_path)
    except Exception:
        return None, None

    # Keep only postseason rows if the column is present
    if 'season_type' in df.columns:
        df = df[df['season_type'].str.lower().str.contains('post', na=False)].copy()

    # Need at minimum scores and net efficiency to derive upset label & features
    required = {'home_score', 'away_score', 'net_eff_diff'}
    if not required.issubset(df.columns):
        return None, None

    df = df.dropna(subset=list(required)).reset_index(drop=True)
    if len(df) < 10:
        return None, None

    # ── Upset label ───────────────────────────────────────────────────────────
    # actual_spread = away_score − home_score  (negative when home wins)
    # We define "favorite" as the team with better net efficiency.
    # Upset = statistically weaker team wins.
    net_eff = df['net_eff_diff'] if 'net_eff_diff' in df.columns else pd.Series(np.zeros(len(df)))
    home_wins  = (df['home_score'] > df['away_score']).astype(int)  # 1 if home wins
    home_better = (net_eff.fillna(0) > 0).astype(int)               # 1 if home has better stats

    # Upset when home is LESS efficient but wins, OR home is MORE efficient but loses
    is_upset = ((home_better == 1) & (home_wins == 0) |
                (home_better == 0) & (home_wins == 1)).astype(int)

    # ── Feature engineering ──────────────────────────────────────────────────
    # For upset model, we want the diff from the *underdog's* perspective:
    # underdog = team with worse efficiency
    # When home is the favorite (net_eff_diff > 0), the away team is the underdog
    #   → underdog net eff diff = −net_eff_diff  (flip sign)
    # When home is the underdog (net_eff_diff <= 0), use as-is
    underdog_net_eff = np.where(home_better == 1,
                                -net_eff.fillna(0).values,
                                net_eff.fillna(0).values)

    # Seed-diff proxy: use net efficiency gap (≈2.5 pts per seed)
    # actual_spread is the game margin, NOT the betting line —  use eff gap instead
    eff_gap = net_eff.abs().fillna(0)
    seed_diff_proxy = (eff_gap / 2.5).clip(upper=15)

    # Tempo difference – use KenPom adj-tempo diff when available
    if 'kenpom_adjt_diff' in df.columns:
        tempo_diff = df['kenpom_adjt_diff'].abs().fillna(0)
    else:
        tempo_diff = pd.Series(np.zeros(len(df)))

    # Underdog defensive efficiency proxy
    if 'def_eff_diff' in df.columns:
        # def_eff_diff = home_def_eff − away_def_eff  (lower = better defense)
        und_def_eff = np.where(home_better == 1,
                               100 - df['def_eff_diff'].fillna(0),   # away (underdog) def eff
                               100 + df['def_eff_diff'].fillna(0))   # home (underdog) def eff
    else:
        und_def_eff = np.full(len(df), 100.0)

    # Conference strength proxy from KenPom SOS when available
    if 'kenpom_sos_diff' in df.columns:
        und_conf = (3 + df['kenpom_sos_diff'].fillna(0) * (-1)).clip(1, 5)
    else:
        und_conf = np.full(len(df), 3.0)

    # ── Assemble feature matrix (14 cols matching create_features output) ─────
    X = np.column_stack([
        seed_diff_proxy.values,          # 1  seed_diff
        underdog_net_eff,                # 2  net_eff_diff
        (underdog_net_eff > 0).astype(int),  # 3 underdog_better_eff
        tempo_diff.values,               # 4  tempo_diff
        np.full(len(df), 0.37),          # 5  underdog_three_rate (neutral default)
        und_def_eff,                     # 6  underdog_def_eff
        np.zeros(len(df)),               # 7  coach_exp_diff (unknown)
        np.full(len(df), 5.0),           # 8  underdog_last_10
        np.full(len(df), 6.0),           # 9  favorite_last_10
        und_conf.values if hasattr(und_conf, 'values') else und_conf,  # 10 underdog_conf_strength
        np.ones(len(df)),                # 11 round_number (first round default)
        np.zeros(len(df)),               # 12 distance_diff
        np.zeros(len(df)),               # 13 underdog_travel_advantage
        np.zeros(len(df)),               # 14 underdog_pseudo_home
    ])

    y = is_upset.values
    return X, y


def create_historical_training_data() -> Tuple[np.array, np.array]:
    """Create synthetic training data based on historical upset patterns.

    Used as a fallback when real CSV data is unavailable.
    """

    # Generate synthetic training data based on historical patterns
    n_samples = 1000
    features = []
    targets = []

    for _ in range(n_samples):
        # Random seed matchup
        fav_seed = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8])
        und_seed = np.random.choice([9, 10, 11, 12, 13, 14, 15, 16])

        # Random locations (simplified - using lat/lon-like coordinates)
        # Tournament sites typically spread across US
        game_location = (np.random.uniform(25, 48), np.random.uniform(-120, -70))
        
        # School locations
        fav_school_location = (np.random.uniform(25, 48), np.random.uniform(-120, -70))
        und_school_location = (np.random.uniform(25, 48), np.random.uniform(-120, -70))

        # Create team stats (simplified)
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
        
        # Game context with location
        game_context = {
            'round_number': np.random.choice([1, 2, 3, 4]),
            'location': game_location
        }

        # Create features
        predictor = UpsetPredictor()
        feature_vector = predictor.create_features(favorite, underdog, game_context)

        # Determine if upset occurred (based on historical rates + factors)
        seed_pair = (fav_seed, und_seed)
        base_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.3)

        # Adjust based on efficiency difference
        eff_diff = underdog['net_efficiency'] - favorite['net_efficiency']
        adjusted_rate = base_rate + (eff_diff * 0.01)  # 1% per efficiency point
        
        # Adjust for location advantage
        from math import sqrt
        def distance(loc1, loc2):
            return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
        
        und_dist = distance(und_school_location, game_location)
        fav_dist = distance(fav_school_location, game_location)
        
        # Closer team gets a small boost
        if und_dist < fav_dist * 0.5:
            adjusted_rate += 0.03  # 3% boost for travel advantage
        if und_dist < 5 and fav_dist > 20:  # Pseudo home game (scaled for coordinates)
            adjusted_rate += 0.05  # 5% boost for home crowd

        # Add some random noise
        upset_occurred = np.random.random() < min(max(adjusted_rate, 0.01), 0.99)

        features.append(feature_vector)
        targets.append(1 if upset_occurred else 0)

    return np.array(features), np.array(targets)


def train_upset_model() -> Tuple['UpsetPredictor', Dict[str, Any]]:
    """Train the upset prediction model and return (predictor, metrics).

    Workflow:
    1. Try to load real historical data from CSV.
    2. Augment with a small batch of synthetic samples for robustness.
    3. Fall back to fully synthetic data only when the CSV is missing/unusable.
    """

    X_real, y_real = create_training_data_from_csv()
    X_syn, y_syn = create_historical_training_data()

    if X_real is not None and len(X_real) >= 50:
        # Blend real data with a small synthetic augmentation
        n_aug = min(200, len(X_syn))
        X = np.vstack([X_real, X_syn[:n_aug]])
        y = np.concatenate([y_real, y_syn[:n_aug]])
        data_source = f"real CSV ({len(X_real)} games) + {n_aug} synthetic"
    else:
        print("Real CSV data not available — falling back to synthetic data.")
        X, y = X_syn, y_syn
        data_source = "synthetic"

    print(f"Training upset model on {len(X)} samples ({data_source})")
    print(f"Upset rate: {y.mean():.1%}")

    predictor = UpsetPredictor()
    results = predictor.train(X, y)
    results['data_source'] = data_source

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


def generate_upset_watch_list(bracket_data: dict,
                              predictor: UpsetPredictor) -> list:
    """Generate list of games with high upset potential."""

    watch_list = []

    for game in bracket_data.get('first_round_games', []):
        favorite = game['favorite']  # Higher seed
        underdog = game['underdog']  # Lower seed

        result = predictor.predict_upset_probability(favorite, underdog)

        if result['upset_probability'] > 0.25:  # 25%+ upset chance
            watch_list.append({
                'matchup': f"({underdog['seed']}) {underdog['name']} vs ({favorite['seed']}) {favorite['name']}",
                'upset_probability': result['upset_probability'],
                'confidence': result['confidence'],
                'key_reasons': result['key_reasons'],
                'historical_rate': result['historical_rate'],
                'round': game.get('round', 'Round of 64')
            })

    # Sort by upset probability
    watch_list.sort(key=lambda x: x['upset_probability'], reverse=True)

    return watch_list


def display_upset_watch(watch_list: list):
    """Display upset watch in a formatted way."""

    print("=" * 60)
    print("🚨 UPSET WATCH LIST")
    print("=" * 60)

    for i, game in enumerate(watch_list, 1):
        print(f"\n{i}. {game['matchup']}")
        print(f"   Upset Probability: {game['upset_probability']:.1%}")
        print(f"   Historical Rate: {game['historical_rate']:.1%}")
        print(f"   Confidence: {game['confidence']}")
        print(f"   Key Factors:")
        for reason in game['key_reasons'][:3]:
            print(f"     • {reason}")


def identify_cinderella_candidates(bracket_data: dict,
                                   simulation_results: dict) -> list:
    """
    Identify potential Cinderella teams (low seeds with deep run potential).
    Cinderella = 10+ seed that reaches Sweet 16 or further.
    """

    def safe_get(d, key, default=0):
        """Safely get value from dict, handling None values."""
        val = d.get(key, default)
        return val if val is not None else default

    candidates = []

    for team_id, probs in simulation_results['team_probabilities'].items():
        seed = safe_get(probs, 'seed', 16)

        # Only consider 10+ seeds
        if seed < 10:
            continue

        sweet_16_prob = safe_get(probs, 'sweet_16_prob', 0)

        # Need at least 10% chance to make Sweet 16
        if sweet_16_prob < 0.10:
            continue

        # Get team stats
        team_stats = bracket_data['teams'].get(team_id, {})

        candidates.append({
            'team': safe_get(probs, 'name', 'Unknown'),
            'seed': seed,
            'region': safe_get(probs, 'region', 'Unknown'),
            'sweet_16_prob': sweet_16_prob,
            'elite_8_prob': safe_get(probs, 'elite_8_prob', 0),
            'final_four_prob': safe_get(probs, 'final_four_prob', 0),
            'net_efficiency': safe_get(team_stats, 'net_efficiency', 0),
            'cinderella_score': calculate_cinderella_score(probs, team_stats)
        })

    # Sort by Cinderella score
    candidates.sort(key=lambda x: x['cinderella_score'], reverse=True)

    return candidates


def calculate_cinderella_score(probs: dict, stats: dict) -> float:
    """
    Calculate a "Cinderella score" combining:
    - Sweet 16 probability
    - Seed (higher = more Cinderella)
    - Efficiency rating
    """

    def safe_get(d, key, default=0):
        """Safely get value from dict, handling None values."""
        val = d.get(key, default)
        return val if val is not None else default

    seed_bonus = (safe_get(probs, 'seed', 10) - 9) * 0.1  # Bonus for higher seeds
    sweet_16_weight = safe_get(probs, 'sweet_16_prob', 0) * 2
    elite_8_weight = safe_get(probs, 'elite_8_prob', 0) * 3
    efficiency_factor = (safe_get(stats, 'net_efficiency', 0) + 20) / 40  # Normalize

    return seed_bonus + sweet_16_weight + elite_8_weight + efficiency_factor


def display_cinderella_candidates(candidates: list):
    """Display potential Cinderella teams."""

    print("=" * 60)
    print("🎃 CINDERELLA CANDIDATES")
    print("(10+ seeds with Sweet 16 potential)")
    print("=" * 60)

    for i, team in enumerate(candidates[:10], 1):
        print(f"\n{i}. ({team['seed']}) {team['team']} - {team['region']}")
        print(f"   Sweet 16: {team['sweet_16_prob']:.1%}")
        print(f"   Elite 8:  {team['elite_8_prob']:.1%}")
        print(f"   Final 4:  {team['final_four_prob']:.1%}")
        print(f"   Cinderella Score: {team['cinderella_score']:.2f}")