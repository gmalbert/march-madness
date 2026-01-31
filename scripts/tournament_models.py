"""Tournament-specific models and utilities.

Implements functions from docs/roadmap-tournament-models.md including:
- train_tournament_model
- create_tournament_features
- train_upset_model
- calculate_upset_features
- adjust_for_round
- apply_seed_prior
- get_historical_matchup_rates
- TournamentPredictor
- evaluation helpers

This module is designed to be independent and robust (safe defaults, lightweight
fallbacks if models are missing).
"""
from typing import Dict, Any, Optional
import numpy as np

try:
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier
    from sklearn.ensemble import GradientBoostingClassifier
except Exception:
    # Allow importing the module even if heavy dependencies are missing.
    XGBClassifier = None
    CalibratedClassifierCV = None
    GradientBoostingClassifier = None


def train_tournament_model(X_regular: np.ndarray, y_regular: np.ndarray,
                           X_tournament: np.ndarray, y_tournament: np.ndarray):
    """Train a tournament-weighted classifier and calibrate probabilities.

    Tournament games are weighted more heavily to emphasize tournament behavior.
    Returns a calibrated classifier when scikit-learn and xgboost are available.
    """
    if XGBClassifier is None or CalibratedClassifierCV is None:
        raise RuntimeError("Required packages (xgboost, sklearn) not available")

    # Combine datasets
    X_combined = np.vstack([X_regular, X_tournament]) if len(X_tournament) else X_regular
    y_combined = np.hstack([y_regular, y_tournament]) if len(y_tournament) else y_regular

    weights = np.ones(len(y_combined))
    if len(y_tournament) > 0:
        weights[len(y_regular):] = 3.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_combined, y_combined, sample_weight=weights)

    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated.fit(X_combined, y_combined, sample_weight=weights)

    return calibrated


def create_tournament_features(team1: Dict[str, Any], team2: Dict[str, Any],
                               game_context: Dict[str, Any]) -> np.ndarray:
    """Create tournament-specific features for a matchup (as numpy array)."""
    features = []

    # Efficiency differentials
    features.append(team1.get('net_efficiency', 0) - team2.get('net_efficiency', 0))
    features.append(team1.get('off_efficiency', 0) - team2.get('off_efficiency', 0))
    features.append(team1.get('def_efficiency', 0) - team2.get('def_efficiency', 0))

    # Seed info
    features.append(team1.get('seed', 16) - team2.get('seed', 16))
    features.append(np.log1p(team1.get('seed', 16)) - np.log1p(team2.get('seed', 16)))

    # Tournament experience
    features.append(team1.get('coach_ncaa_wins', 0) - team2.get('coach_ncaa_wins', 0))
    features.append(team1.get('program_ff_apps', 0) - team2.get('program_ff_apps', 0))

    # Momentum
    features.append(team1.get('conf_tourney_wins', 0) - team2.get('conf_tourney_wins', 0))
    features.append(team1.get('last_10_wins', 5) - team2.get('last_10_wins', 5))

    # Style matchups
    features.append(team1.get('tempo', 70) - team2.get('tempo', 70))
    features.append(team1.get('three_rate', 0.35) - team2.get('three_rate', 0.35))
    features.append(team1.get('ft_rate', 0.30) - team2.get('ft_rate', 0.30))

    # Game context
    features.append(game_context.get('round_number', 1))
    features.append(game_context.get('travel_advantage', 0))

    return np.array(features, dtype=float)


def train_upset_model(X: np.ndarray, y_upset: np.ndarray):
    """Train a model specialized to predict upset probability."""
    if GradientBoostingClassifier is None:
        raise RuntimeError("sklearn is required to train upset model")

    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        random_state=42
    )
    model.fit(X, y_upset)
    return model


def calculate_upset_features(higher_seed_team: Dict[str, Any], lower_seed_team: Dict[str, Any]) -> np.ndarray:
    """Build features to detect upset potential (lower_seed vs higher_seed).

    Note: lower_seed_team is the team with the higher numerical seed (worse seed).
    """
    features = []

    seed_diff = lower_seed_team.get('seed', 16) - higher_seed_team.get('seed', 1)
    features.append(seed_diff)

    eff_diff = lower_seed_team.get('net_efficiency', 0) - higher_seed_team.get('net_efficiency', 0)
    features.append(eff_diff)

    features.append(1 if eff_diff > 0 else 0)

    tempo_diff = abs(higher_seed_team.get('tempo', 70) - lower_seed_team.get('tempo', 70))
    features.append(tempo_diff)

    features.append(lower_seed_team.get('three_rate', 0.35))
    features.append(higher_seed_team.get('three_rate', 0.35))

    features.append(lower_seed_team.get('def_efficiency', 100))

    features.append(higher_seed_team.get('injury_impact', 0))

    return np.array(features, dtype=float)


def adjust_for_round(base_probability: float, round_number: int,
                      team1_seed: int, team2_seed: int) -> float:
    """Adjust win probability based on tournament round (favoring favorites in later rounds)."""
    round_chalk_factor = {
        1: 1.00,
        2: 1.02,
        3: 1.05,
        4: 1.08,
        5: 1.10,
        6: 1.12
    }
    factor = round_chalk_factor.get(round_number, 1.0)

    if base_probability > 0.5:
        adjusted = base_probability * factor
        return min(adjusted, 0.95)
    else:
        adjustment = 2 - factor
        adjusted = base_probability * adjustment
        return max(adjusted, 0.05)


def get_historical_matchup_rates() -> Dict[tuple, float]:
    """Return historical win rates for seed matchups (higher seed perspective)."""
    return {
        (1, 16): 0.99,
        (2, 15): 0.94,
        (3, 14): 0.85,
        (4, 13): 0.79,
        (5, 12): 0.65,
        (6, 11): 0.63,
        (7, 10): 0.61,
        (8, 9): 0.51,
        (1, 8): 0.80, (1, 9): 0.85,
        (4, 5): 0.55, (3, 6): 0.60,
        (1, 4): 0.70, (1, 5): 0.75,
        (2, 3): 0.55, (2, 6): 0.65,
        (1, 2): 0.55, (1, 3): 0.60,
    }


def apply_seed_prior(model_probability: float, seed1: int, seed2: int, prior_weight: float = 0.2) -> float:
    """Blend model probability with historical seed performance."""
    historical_rates = get_historical_matchup_rates()
    matchup_key = (min(seed1, seed2), max(seed1, seed2))
    historical_prob = historical_rates.get(matchup_key, 0.5)

    # If seed1 is the higher seed (worse number), flip perspective
    if seed1 > seed2:
        historical_prob = 1 - historical_prob

    blended = (1 - prior_weight) * model_probability + prior_weight * historical_prob
    return float(blended)


class TournamentPredictor:
    """Ensemble model for tournament predictions.

    If base_models are provided they should be a dict with 'moneyline' models
    that expose predict_proba. Upset and tournament models can be set later.
    """
    def __init__(self, base_models: Optional[Dict] = None):
        self.base_models = base_models or {}
        self.tournament_model = None
        self.upset_model = None

    def train(self, X_regular, y_regular, X_tournament, y_tournament, X_upset=None, y_upset=None):
        if XGBClassifier is None or CalibratedClassifierCV is None or GradientBoostingClassifier is None:
            raise RuntimeError("Required ML packages not available for training")

        self.tournament_model = train_tournament_model(X_regular, y_regular, X_tournament, y_tournament)
        if X_upset is not None and y_upset is not None:
            self.upset_model = train_upset_model(X_upset, y_upset)

    def predict_game(self, team1: Dict[str, Any], team2: Dict[str, Any], game_context: Dict[str, Any]) -> Dict[str, Any]:
        # Base model predictions (average across base models if available)
        base_probs = []
        if self.base_models.get('moneyline'):
            for name, model in self.base_models['moneyline'].items():
                try:
                    prob = model.predict_proba(self._create_base_features(team1, team2))[0][1]
                    base_probs.append(prob)
                except Exception:
                    continue
        base_prob = float(np.mean(base_probs)) if base_probs else 0.5

        tournament_features = create_tournament_features(team1, team2, game_context).reshape(1, -1)
        tournament_prob = float(self.tournament_model.predict_proba(tournament_features)[0][1]) if self.tournament_model else base_prob

        # Upset risk
        seed1, seed2 = team1.get('seed', 16), team2.get('seed', 16)
        upset_risk = 0
        if self.upset_model is not None:
            if seed1 < seed2:
                upset_features = calculate_upset_features(team1, team2).reshape(1, -1)
                upset_risk = float(self.upset_model.predict_proba(upset_features)[0][1])
            elif seed2 < seed1:
                upset_features = calculate_upset_features(team2, team1).reshape(1, -1)
                upset_risk = float(self.upset_model.predict_proba(upset_features)[0][1])

        combined_prob = 0.4 * base_prob + 0.4 * tournament_prob
        combined_prob = apply_seed_prior(combined_prob, seed1, seed2, prior_weight=0.2)

        round_num = game_context.get('round_number', 1)
        final_prob = adjust_for_round(combined_prob, round_num, seed1, seed2)

        return {
            'team1_win_prob': float(final_prob),
            'team2_win_prob': float(1 - final_prob),
            'predicted_winner': team1.get('name') if final_prob > 0.5 else team2.get('name'),
            'confidence': float(max(final_prob, 1 - final_prob)),
            'upset_risk': float(upset_risk),
            'is_upset_prediction': (final_prob < 0.5 and seed1 < seed2) or (final_prob > 0.5 and seed2 < seed1),
            'components': {
                'base_model': base_prob,
                'tournament_model': tournament_prob,
                'seed_adjusted': combined_prob,
                'round_adjusted': final_prob
            }
        }

    def _create_base_features(self, team1: Dict[str, Any], team2: Dict[str, Any]) -> np.ndarray:
        return np.array([[
            team1.get('off_efficiency', 0) - team2.get('off_efficiency', 0),
            team1.get('def_efficiency', 0) - team2.get('def_efficiency', 0),
            team1.get('net_efficiency', 0) - team2.get('net_efficiency', 0)
        ]])


# Evaluation helpers
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss


def evaluate_tournament_model(predictions: list, actuals: list) -> dict:
    """Evaluate tournament predictions (accuracy, brier, log loss, upset metrics)."""
    y_pred = [1 if p['team1_win_prob'] > 0.5 else 0 for p in predictions]
    y_prob = [p['team1_win_prob'] for p in predictions]
    y_true = actuals

    # By-round accuracy
    round_accuracy = {}
    rounds = [pred.get('round', 1) for pred in predictions]
    for round_num in sorted(set(rounds)):
        idx = [i for i, r in enumerate(rounds) if r == round_num]
        if idx:
            round_accuracy[round_num] = accuracy_score([y_true[i] for i in idx], [y_pred[i] for i in idx])

    upset_predictions = [p for p in predictions if p.get('is_upset_prediction')]
    upset_correct = 0
    for p, a in zip(predictions, actuals):
        if p.get('is_upset_prediction'):
            # Upset predicted when favorite loses
            if (p['team1_win_prob'] < 0.5 and a == 0) or (p['team1_win_prob'] > 0.5 and a == 1):
                upset_correct += 1

    return {
        'overall_accuracy': accuracy_score(y_true, y_pred),
        'brier_score': brier_score_loss(y_true, y_prob),
        'log_loss': log_loss(y_true, y_prob),
        'round_accuracy': round_accuracy,
        'upset_predictions': len(upset_predictions),
        'upset_accuracy': upset_correct / max(len(upset_predictions), 1),
        'bracket_score': calculate_bracket_score(predictions, actuals)
    }


def calculate_bracket_score(predictions: list, actuals: list) -> int:
    ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
    score = 0
    for pred, actual in zip(predictions, actuals):
        predicted_correct = (pred['team1_win_prob'] > 0.5) == (actual == 1)
        if predicted_correct:
            score += ROUND_POINTS.get(pred.get('round', 1), 10)
    return score
