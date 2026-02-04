# Roadmap: Tournament Prediction Models

*Model architecture for predicting tournament game outcomes.*

## Status Summary

**✅ MOSTLY IMPLEMENTED** - Tournament model architecture is complete, but not integrated into main prediction pipeline.

- ✅ **Tournament Model Functions**: All core functions implemented in `scripts/tournament_models.py`
- ✅ **Upset Detection**: Implemented in `upset_prediction.py` and `tournament_models.py`
- ✅ **Bracket Simulation**: Monte Carlo simulation implemented in `bracket_simulation.py`
- ✅ **Tournament CV**: Cross-validation implemented in `advanced_model_training.py`
- ❌ **Integration**: Tournament models are NOT used in main prediction pipeline (`generate_predictions.py` uses only regular season models)

## Model Strategy

### Why Tournament Models Differ from Regular Season

```python
# Tournament games are different:
TOURNAMENT_FACTORS = {
    'single_elimination': 'No room for error - teams play differently',
    'neutral_sites': 'No true home court advantage (mostly)',
    'pressure': 'Experience and coaching matter more',
    'preparation': 'Teams have days to prepare for specific opponent',
    'matchups': 'Style matchups become critical',
    'variance': 'March Madness - upsets are more common than expected'
}
```

## Model Architecture

### 1. Base Win Probability Model (Already Have)

```python
# Use existing model as foundation
from pathlib import Path
import joblib

def load_base_models():
    """Load trained regular season models."""
    MODEL_DIR = Path("data_files/models")
    
    return {
        'moneyline': {
            'xgboost': joblib.load(MODEL_DIR / 'moneyline_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'moneyline_random_forest.joblib'),
            'logistic': joblib.load(MODEL_DIR / 'moneyline_logistic_regression.joblib')
        },
        'spread': {
            'xgboost': joblib.load(MODEL_DIR / 'spread_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'spread_random_forest.joblib'),
        }
    }
```

### 2. Tournament-Adjusted Model

```python
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from xgboost import XGBClassifier

def train_tournament_model(X_regular, y_regular, X_tournament, y_tournament):
    """
    Train a model that combines regular season and tournament data.
    Tournament data is weighted more heavily.
    """ - ✅ IMPLEMENTED in scripts/tournament_models.py
    from sklearn.utils.class_weight import compute_sample_weight
    
    # Combine datasets with tournament games weighted 3x
    X_combined = np.vstack([X_regular, X_tournament])
    y_combined = np.hstack([y_regular, y_tournament])
    
    # Weight tournament games more heavily
    weights = np.ones(len(y_combined))
    weights[len(y_regular):] = 3.0  # Tournament games weighted 3x
    
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
    
    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=5)
    calibrated.fit(X_combined, y_combined, sample_weight=weights)
    
    return calibrated


def create_tournament_features(team1: dict, team2: dict, game_context: dict) -> np.array:
    """Create feature vector for tournament game prediction.""" - ✅ IMPLEMENTED in scripts/tournament_models.py
    
    features = []
    
    # Efficiency differentials (core features)
    features.append(team1['net_efficiency'] - team2['net_efficiency'])
    features.append(team1['off_efficiency'] - team2['off_efficiency'])
    features.append(team1['def_efficiency'] - team2['def_efficiency'])
    
    # Seed information
    features.append(team1['seed'] - team2['seed'])
    features.append(np.log1p(team1['seed']) - np.log1p(team2['seed']))
    
    # Tournament experience
    features.append(team1.get('coach_ncaa_wins', 0) - team2.get('coach_ncaa_wins', 0))
    features.append(team1.get('program_ff_apps', 0) - team2.get('program_ff_apps', 0))
    
    # Momentum (conference tournament + last 10)
    features.append(team1.get('conf_tourney_wins', 0) - team2.get('conf_tourney_wins', 0))
    features.append(team1.get('last_10_wins', 5) - team2.get('last_10_wins', 5))
    
    # Style matchup features
    features.append(team1.get('tempo', 70) - team2.get('tempo', 70))
    features.append(team1.get('three_rate', 0.35) - team2.get('three_rate', 0.35))
    features.append(team1.get('ft_rate', 0.30) - team2.get('ft_rate', 0.30))
    
    # Game context
    features.append(game_context.get('round_number', 1))  # 1-6
    features.append(game_context.get('travel_advantage', 0))  # Distance differential
    
    return np.array(features)
```

### 3. Upset Probability Model

```python
def train_upset_model(X, y_upset):
    """
    Specialized model for predicting upsets.
    y_upset = 1 if lower seed wins, 0 otherwise.
    """ - ✅ IMPLEMENTED in scripts/tournament_models.py
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Class weights to handle imbalanced data (upsets are minority)
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        random_state=42
    )
    
    model.fit(X, y_upset)
    return model


def calculate_upset_features(higher_seed_team: dict, lower_seed_team: dict) -> np.array:
    """Features specifically designed to detect upset potential.""" - ✅ IMPLEMENTED in scripts/tournament_models.py
    
    features = []
    
    # Seed differential (smaller = more likely upset)
    seed_diff = lower_seed_team['seed'] - higher_seed_team['seed']
    features.append(seed_diff)
    
    # Efficiency "wrongness" - is the lower seed actually better?
    eff_diff = lower_seed_team['net_efficiency'] - higher_seed_team['net_efficiency']
    features.append(eff_diff)
    features.append(1 if eff_diff > 0 else 0)  # Lower seed has better efficiency
    
    # Tempo mismatch (can cause upsets)
    tempo_diff = abs(higher_seed_team['tempo'] - lower_seed_team['tempo'])
    features.append(tempo_diff)
    
    # Three-point reliance (high variance = upset potential)
    features.append(lower_seed_team.get('three_rate', 0.35))
    features.append(higher_seed_team.get('three_rate', 0.35))
    
    # Defense quality of lower seed
    features.append(lower_seed_team.get('def_efficiency', 100))
    
    # Higher seed injuries/issues (would need external data)
    features.append(higher_seed_team.get('injury_impact', 0))
    
    return np.array(features)
```

### 4. Round-Specific Adjustments

```python
def adjust_for_round(base_probability: float, round_number: int, 
                      team1_seed: int, team2_seed: int) -> float:
    """
    Adjust win probability based on tournament round.
    Later rounds have more chalk (favorites win more).
    """ - ✅ IMPLEMENTED in scripts/tournament_models.py
    
    # Round adjustments based on historical data
    round_chalk_factor = {
        1: 1.00,  # Round of 64 - most upsets
        2: 1.02,  # Round of 32
        3: 1.05,  # Sweet 16
        4: 1.08,  # Elite 8
        5: 1.10,  # Final Four
        6: 1.12   # Championship
    }
    
    # If favorite (higher probability), increase slightly in later rounds
    if base_probability > 0.5:
        adjustment = round_chalk_factor.get(round_number, 1.0)
        adjusted = base_probability * adjustment
        # Don't exceed reasonable bounds
        return min(adjusted, 0.95)
    else:
        # Underdog - reduce probability slightly in later rounds
        adjustment = 2 - round_chalk_factor.get(round_number, 1.0)
        adjusted = base_probability * adjustment
        return max(adjusted, 0.05)


def apply_seed_prior(model_probability: float, seed1: int, seed2: int,
                     prior_weight: float = 0.2) -> float:
    """
    Blend model probability with historical seed performance.
    """ - ✅ IMPLEMENTED in scripts/tournament_models.py
    # Historical win rates for seed matchups
    historical_rates = get_historical_matchup_rates()
    
    matchup_key = (min(seed1, seed2), max(seed1, seed2))
    historical_prob = historical_rates.get(matchup_key, 0.5)
    
    # If seed1 is the higher seed, flip the probability
    if seed1 > seed2:
        historical_prob = 1 - historical_prob
    
    # Weighted blend
    blended = (1 - prior_weight) * model_probability + prior_weight * historical_prob
    
    return blended

def get_historical_matchup_rates() -> dict:
    """Historical win rates for seed matchups (higher seed perspective).""" - ✅ IMPLEMENTED in scripts/tournament_models.py
    return {
        (1, 16): 0.99,
        (2, 15): 0.94,
        (3, 14): 0.85,
        (4, 13): 0.79,
        (5, 12): 0.65,
        (6, 11): 0.63,
        (7, 10): 0.61,
        (8, 9): 0.51,
        # Later rounds
        (1, 8): 0.80, (1, 9): 0.85,
        (4, 5): 0.55, (3, 6): 0.60,
        (1, 4): 0.70, (1, 5): 0.75,
        (2, 3): 0.55, (2, 6): 0.65,
        (1, 2): 0.55, (1, 3): 0.60,
    }
```

## Ensemble Tournament Model

```python
class TournamentPredictor:
    """Ensemble model for tournament predictions.""" - ✅ IMPLEMENTED in scripts/tournament_models.py
    
    def __init__(self):
        self.base_models = load_base_models()
        self.tournament_model = None
        self.upset_model = None
        
    def train(self, X_regular, y_regular, X_tournament, y_tournament, X_upset, y_upset):
        """Train all component models."""
        self.tournament_model = train_tournament_model(
            X_regular, y_regular, X_tournament, y_tournament
        )
        self.upset_model = train_upset_model(X_upset, y_upset)
    
    def predict_game(self, team1: dict, team2: dict, 
                     game_context: dict) -> dict:
        """Predict a single tournament game."""
        
        # Get base model prediction
        base_features = self._create_base_features(team1, team2)
        base_probs = []
        for model_type, models in self.base_models['moneyline'].items():
            try:
                prob = models.predict_proba(base_features)[0][1]
                base_probs.append(prob)
            except:
                continue
        base_prob = np.mean(base_probs) if base_probs else 0.5
        
        # Get tournament model prediction
        tournament_features = create_tournament_features(team1, team2, game_context)
        tournament_prob = self.tournament_model.predict_proba(
            tournament_features.reshape(1, -1)
        )[0][1]
        
        # Get upset probability (if applicable)
        seed1, seed2 = team1['seed'], team2['seed']
        upset_risk = 0
        if seed1 < seed2:  # team1 is favorite
            upset_features = calculate_upset_features(team1, team2)
            upset_risk = self.upset_model.predict_proba(
                upset_features.reshape(1, -1)
            )[0][1]
        elif seed2 < seed1:  # team2 is favorite
            upset_features = calculate_upset_features(team2, team1)
            upset_risk = self.upset_model.predict_proba(
                upset_features.reshape(1, -1)
            )[0][1]
        
        # Combine predictions
        # Weight: 40% base, 40% tournament, 20% seed prior
        combined_prob = 0.4 * base_prob + 0.4 * tournament_prob
        combined_prob = apply_seed_prior(combined_prob, seed1, seed2, prior_weight=0.2)
        
        # Apply round adjustment
        round_num = game_context.get('round_number', 1)
        final_prob = adjust_for_round(combined_prob, round_num, seed1, seed2)
        
        return {
            'team1_win_prob': final_prob,
            'team2_win_prob': 1 - final_prob,
            'predicted_winner': team1['name'] if final_prob > 0.5 else team2['name'],
            'confidence': max(final_prob, 1 - final_prob),
            'upset_risk': upset_risk,
            'is_upset_prediction': (final_prob < 0.5 and seed1 < seed2) or 
                                   (final_prob > 0.5 and seed2 < seed1),
            'components': {
                'base_model': base_prob,
                'tournament_model': tournament_prob,
                'seed_adjusted': combined_prob,
                'round_adjusted': final_prob
            }
        }
    
    def _create_base_features(self, team1: dict, team2: dict) -> np.array:
        """Create features for base models."""
        return np.array([[
            team1['off_efficiency'] - team2['off_efficiency'],
            team1['def_efficiency'] - team2['def_efficiency'],
            team1['net_efficiency'] - team2['net_efficiency']
        ]])
```

## Model Evaluation

```python
def evaluate_tournament_model(predictions: list, actuals: list) -> dict:
    """Evaluate tournament prediction accuracy.""" - ✅ IMPLEMENTED in scripts/tournament_models.py
    from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
    
    # Extract predictions and probabilities
    y_pred = [1 if p['team1_win_prob'] > 0.5 else 0 for p in predictions]
    y_prob = [p['team1_win_prob'] for p in predictions]
    y_true = actuals
    
    # By-round accuracy
    round_accuracy = {}
    for round_num in range(1, 7):
        round_preds = [(p, a) for p, a, r in zip(y_pred, y_true, 
                       [pred['round'] for pred in predictions]) if r == round_num]
        if round_preds:
            round_accuracy[round_num] = accuracy_score(
                [a for _, a in round_preds],
                [p for p, _ in round_preds]
            )
    
    # Upset detection
    upset_predictions = [p for p in predictions if p.get('is_upset_prediction')]
    upset_correct = sum(1 for p, a in zip(predictions, actuals) 
                        if p.get('is_upset_prediction') and 
                        ((p['team1_win_prob'] < 0.5) == (a == 0)))
    
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
    """Calculate ESPN-style bracket score.""" - ✅ IMPLEMENTED in scripts/tournament_models.py
    ROUND_POINTS = {1: 10, 2: 20, 3: 40, 4: 80, 5: 160, 6: 320}
    
    score = 0
    for pred, actual in zip(predictions, actuals):
        predicted_correct = (pred['team1_win_prob'] > 0.5) == (actual == 1)
        if predicted_correct:
            score += ROUND_POINTS.get(pred.get('round', 1), 10)
    
    return score
```

## Next Steps

1. Collect historical tournament data for training - ✅ PARTIALLY IMPLEMENTED (tournament CV exists in advanced_model_training.py)
2. Train tournament-specific models - ❌ NOT IMPLEMENTED (tournament models exist but are not trained or used)
3. Integrate tournament models into main prediction pipeline - ❌ NOT IMPLEMENTED (generate_predictions.py only uses regular season models)
4. See `roadmap-bracket-simulation.md` for Monte Carlo simulation - ✅ IMPLEMENTED (bracket_simulation.py)
5. See `roadmap-upset-detection.md` for upset analysis - ✅ IMPLEMENTED (upset_prediction.py and tournament_models.py)
