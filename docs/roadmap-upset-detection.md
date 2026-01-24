# Roadmap: Upset Detection

*Identifying and predicting tournament upsets.*

## What Makes an Upset?

```python
# Definition: Lower seed defeats higher seed
def is_upset(winner_seed: int, loser_seed: int) -> bool:
    """An upset occurs when the higher-seeded team loses."""
    return winner_seed > loser_seed

# Upset magnitude
def upset_magnitude(winner_seed: int, loser_seed: int) -> int:
    """How big is the upset? Larger = more surprising."""
    return winner_seed - loser_seed if is_upset(winner_seed, loser_seed) else 0

# Historical upset rates by seed matchup
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
```

## Upset Indicators

### 1. Efficiency Mismatch

```python
def check_efficiency_mismatch(higher_seed: dict, lower_seed: dict) -> dict:
    """
    Check if lower seed is actually better by efficiency metrics.
    This is the #1 upset indicator.
    """
    
    result = {
        'net_eff_favors_underdog': False,
        'off_eff_favors_underdog': False,
        'def_eff_favors_underdog': False,
        'upset_indicator_strength': 0.0
    }
    
    # Net efficiency comparison
    net_diff = lower_seed['net_efficiency'] - higher_seed['net_efficiency']
    if net_diff > 0:
        result['net_eff_favors_underdog'] = True
        result['upset_indicator_strength'] += 0.4
    
    # Offensive efficiency
    off_diff = lower_seed['off_efficiency'] - higher_seed['off_efficiency']
    if off_diff > 2:
        result['off_eff_favors_underdog'] = True
        result['upset_indicator_strength'] += 0.2
    
    # Defensive efficiency (lower is better)
    def_diff = higher_seed['def_efficiency'] - lower_seed['def_efficiency']
    if def_diff > 2:
        result['def_eff_favors_underdog'] = True
        result['upset_indicator_strength'] += 0.2
    
    return result
```

### 2. Style Matchup Analysis

```python
def analyze_style_matchup(favorite: dict, underdog: dict) -> dict:
    """
    Analyze if the underdog's style matches up well against the favorite.
    """
    
    factors = {
        'tempo_advantage': False,
        'three_point_variance': False,
        'defensive_disruption': False,
        'upset_probability_boost': 0.0
    }
    
    # Tempo mismatch - if underdog wants to slow down a fast team
    # or speed up a slow team, they can disrupt
    tempo_diff = abs(favorite['tempo'] - underdog['tempo'])
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
```

### 3. Experience & Momentum

```python
def analyze_experience_momentum(favorite: dict, underdog: dict) -> dict:
    """
    Experience and recent form matter in tournament.
    """
    
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
```

### 4. Location & Travel

```python
def analyze_location_advantage(favorite: dict, underdog: dict, 
                               game_location: tuple) -> dict:
    """
    Neutral sites aren't always neutral - proximity matters.
    """
    from math import sqrt
    
    def distance(loc1: tuple, loc2: tuple) -> float:
        """Simple Euclidean distance (would use haversine for real)."""
        return sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    factors = {
        'travel_advantage': False,
        'pseudo_home_game': False,
        'upset_probability_boost': 0.0
    }
    
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
```

## Upset Prediction Model

```python
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

class UpsetPredictor:
    """Specialized model for predicting upsets."""
    
    def __init__(self):
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_split=20,
            subsample=0.8,
            random_state=42
        )
        self.feature_names = []
    
    def create_features(self, favorite: dict, underdog: dict, 
                        game_context: dict = None) -> np.array:
        """Create feature vector for upset prediction."""
        
        features = []
        self.feature_names = []
        
        # Seed differential (smaller = more likely upset)
        seed_diff = underdog['seed'] - favorite['seed']
        features.append(seed_diff)
        self.feature_names.append('seed_diff')
        
        # Efficiency metrics
        net_eff_diff = underdog['net_efficiency'] - favorite['net_efficiency']
        features.append(net_eff_diff)
        self.feature_names.append('net_eff_diff')
        
        features.append(1 if net_eff_diff > 0 else 0)
        self.feature_names.append('underdog_better_eff')
        
        # Style factors
        tempo_diff = abs(favorite.get('tempo', 70) - underdog.get('tempo', 70))
        features.append(tempo_diff)
        self.feature_names.append('tempo_diff')
        
        features.append(underdog.get('three_rate', 0.35))
        self.feature_names.append('underdog_three_rate')
        
        features.append(underdog.get('def_efficiency', 100))
        self.feature_names.append('underdog_def_eff')
        
        # Experience
        features.append(underdog.get('coach_ncaa_games', 0) - 
                       favorite.get('coach_ncaa_games', 0))
        self.feature_names.append('coach_exp_diff')
        
        # Momentum
        features.append(underdog.get('last_10_wins', 5))
        self.feature_names.append('underdog_last_10')
        
        features.append(favorite.get('last_10_wins', 5))
        self.feature_names.append('favorite_last_10')
        
        # Conference strength (using as proxy for schedule strength)
        features.append(underdog.get('conf_strength', 5))
        self.feature_names.append('underdog_conf_strength')
        
        # Round number (upsets less common in later rounds)
        round_num = game_context.get('round_number', 1) if game_context else 1
        features.append(round_num)
        self.feature_names.append('round_number')
        
        return np.array(features)
    
    def train(self, X: np.array, y: np.array):
        """Train the upset prediction model."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'feature_importance': dict(zip(
                self.feature_names,
                self.model.feature_importances_
            ))
        }
    
    def predict_upset_probability(self, favorite: dict, underdog: dict,
                                  game_context: dict = None) -> dict:
        """Predict probability of an upset."""
        
        features = self.create_features(favorite, underdog, game_context)
        
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        upset_prob = proba[1]  # Probability of upset
        
        # Analyze contributing factors
        efficiency = check_efficiency_mismatch(favorite, underdog)
        style = analyze_style_matchup(favorite, underdog)
        experience = analyze_experience_momentum(favorite, underdog)
        
        return {
            'upset_probability': upset_prob,
            'confidence': 'High' if upset_prob > 0.4 else 'Medium' if upset_prob > 0.25 else 'Low',
            'factors': {
                'efficiency_mismatch': efficiency,
                'style_matchup': style,
                'experience_momentum': experience
            },
            'key_reasons': self._get_key_reasons(efficiency, style, experience),
            'historical_rate': HISTORICAL_UPSET_RATES.get(
                (favorite['seed'], underdog['seed']), 0.3
            )
        }
    
    def _get_key_reasons(self, efficiency: dict, style: dict, 
                         experience: dict) -> list:
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
```

## Upset Watch List

```python
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
```

## Cinderella Detection

```python
def identify_cinderella_candidates(bracket_data: dict,
                                   simulation_results: dict) -> list:
    """
    Identify potential Cinderella teams (low seeds with deep run potential).
    Cinderella = 10+ seed that reaches Sweet 16 or further.
    """
    
    candidates = []
    
    for team_id, probs in simulation_results['team_probabilities'].items():
        seed = probs['seed']
        
        # Only consider 10+ seeds
        if seed < 10:
            continue
        
        sweet_16_prob = probs['sweet_16_prob']
        
        # Need at least 10% chance to make Sweet 16
        if sweet_16_prob < 0.10:
            continue
        
        # Get team stats
        team_stats = bracket_data['teams'].get(team_id, {})
        
        candidates.append({
            'team': probs['name'],
            'seed': seed,
            'region': probs['region'],
            'sweet_16_prob': sweet_16_prob,
            'elite_8_prob': probs['elite_8_prob'],
            'final_four_prob': probs['final_four_prob'],
            'net_efficiency': team_stats.get('net_efficiency', 0),
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
    
    seed_bonus = (probs['seed'] - 9) * 0.1  # Bonus for higher seeds
    sweet_16_weight = probs['sweet_16_prob'] * 2
    elite_8_weight = probs['elite_8_prob'] * 3
    efficiency_factor = (stats.get('net_efficiency', 0) + 20) / 40  # Normalize
    
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
```

## Streamlit Integration

```python
import streamlit as st

def render_upset_section(bracket_data: dict, predictor: UpsetPredictor,
                         simulation_results: dict):
    """Render upset analysis section in Streamlit."""
    
    st.header("🚨 Upset Watch")
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["Upset Predictions", "Cinderella Candidates"])
    
    with tab1:
        watch_list = generate_upset_watch_list(bracket_data, predictor)
        
        for game in watch_list[:10]:
            with st.expander(f"⚠️ {game['matchup']} - {game['upset_probability']:.1%} upset chance"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Upset Probability", f"{game['upset_probability']:.1%}")
                    st.metric("Historical Rate", f"{game['historical_rate']:.1%}")
                
                with col2:
                    st.write("**Key Factors:**")
                    for reason in game['key_reasons']:
                        st.write(f"• {reason}")
    
    with tab2:
        candidates = identify_cinderella_candidates(bracket_data, simulation_results)
        
        st.write("Teams with the best chance to make a deep run as a 10+ seed:")
        
        for team in candidates[:8]:
            with st.expander(f"🎃 ({team['seed']}) {team['team']}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Sweet 16", f"{team['sweet_16_prob']:.1%}")
                with col2:
                    st.metric("Elite 8", f"{team['elite_8_prob']:.1%}")
                with col3:
                    st.metric("Final Four", f"{team['final_four_prob']:.1%}")
```

## Next Steps

1. Collect historical upset data for training
2. Train upset prediction model
3. Integrate with main bracket simulation
4. See `roadmap-implementation.md` for full integration plan
