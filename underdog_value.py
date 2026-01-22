"""
Underdog Value Bet Identification

This module identifies underdog betting opportunities where the model predicts
a higher probability of winning than the betting odds suggest.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import joblib

# Configuration
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"


def moneyline_to_implied_probability(moneyline: int) -> float:
    """Convert American moneyline odds to implied probability.
    
    Args:
        moneyline: American odds (e.g., -150, +200)
        
    Returns:
        Implied probability as a decimal (0-1)
    """
    if moneyline < 0:
        # Favorite
        return abs(moneyline) / (abs(moneyline) + 100)
    else:
        # Underdog
        return 100 / (moneyline + 100)


def implied_probability_to_moneyline(probability: float) -> int:
    """Convert implied probability to American moneyline odds.
    
    Args:
        probability: Decimal probability (0-1)
        
    Returns:
        American odds
    """
    if probability >= 0.5:
        # Favorite
        return int(-probability / (1 - probability) * 100)
    else:
        # Underdog
        return int((1 - probability) / probability * 100)


def calculate_expected_value(
    win_probability: float,
    moneyline: int,
    bet_amount: float = 100
) -> Tuple[float, float]:
    """Calculate expected value (EV) for a moneyline bet.
    
    Args:
        win_probability: Model's predicted probability of winning (0-1)
        moneyline: American moneyline odds
        bet_amount: Amount to bet (default $100)
        
    Returns:
        Tuple of (expected_value, roi_percentage)
    """
    if moneyline < 0:
        # Favorite odds
        payout = bet_amount * (100 / abs(moneyline))
    else:
        # Underdog odds
        payout = bet_amount * (moneyline / 100)
    
    # EV = (Probability of Win × Profit) - (Probability of Loss × Stake)
    expected_value = (win_probability * payout) - ((1 - win_probability) * bet_amount)
    roi = (expected_value / bet_amount) * 100
    
    return expected_value, roi


def identify_underdog_value(
    game: Dict,
    home_win_prob: float,
    min_ev_threshold: float = 5.0,
    min_underdog_prob: float = 0.30
) -> Optional[Dict]:
    """Identify if there's value in betting on an underdog.
    
    Args:
        game: Game dictionary with betting lines
        home_win_prob: Model's predicted home team win probability
        min_ev_threshold: Minimum expected value % to consider (default 5%)
        min_underdog_prob: Minimum probability for underdog to consider (default 30%)
        
    Returns:
        Dictionary with value bet info if found, None otherwise
    """
    home_ml = game.get('home_moneyline')
    away_ml = game.get('away_moneyline')
    
    if not home_ml or not away_ml:
        return None
    
    # Calculate implied probabilities from odds
    home_implied = moneyline_to_implied_probability(home_ml)
    away_implied = moneyline_to_implied_probability(away_ml)
    
    # Remove vig (normalize probabilities to sum to 1)
    total_prob = home_implied + away_implied
    home_implied_true = home_implied / total_prob
    away_implied_true = away_implied / total_prob
    
    # Model probabilities
    away_win_prob = 1 - home_win_prob
    
    value_bets = []
    
    # Check home team value (if underdog)
    if home_ml > 0 and home_win_prob >= min_underdog_prob:
        ev, roi = calculate_expected_value(home_win_prob, home_ml)
        if roi >= min_ev_threshold:
            value_bets.append({
                'team': game.get('home_team', 'Home'),
                'side': 'home',
                'model_prob': home_win_prob,
                'implied_prob': home_implied_true,
                'edge': home_win_prob - home_implied_true,
                'moneyline': home_ml,
                'expected_value': ev,
                'roi': roi,
                'bet_type': 'underdog'
            })
    
    # Check away team value (if underdog)
    if away_ml > 0 and away_win_prob >= min_underdog_prob:
        ev, roi = calculate_expected_value(away_win_prob, away_ml)
        if roi >= min_ev_threshold:
            value_bets.append({
                'team': game.get('away_team', 'Away'),
                'side': 'away',
                'model_prob': away_win_prob,
                'implied_prob': away_implied_true,
                'edge': away_win_prob - away_implied_true,
                'moneyline': away_ml,
                'expected_value': ev,
                'roi': roi,
                'bet_type': 'underdog'
            })
    
    # Return the best value bet if any
    if value_bets:
        return max(value_bets, key=lambda x: x['roi'])
    
    return None


def analyze_all_games_for_value(
    games: List[Dict],
    predictions: List[Dict],
    min_ev_threshold: float = 5.0
) -> pd.DataFrame:
    """Analyze all games and identify value betting opportunities.
    
    Args:
        games: List of game dictionaries with betting lines
        predictions: List of prediction dictionaries with win probabilities
        min_ev_threshold: Minimum EV% threshold (default 5%)
        
    Returns:
        DataFrame of value betting opportunities sorted by ROI
    """
    value_opportunities = []
    
    for game, pred in zip(games, predictions):
        if not pred.get('moneyline'):
            continue
            
        home_win_prob = pred['moneyline'].get('home_win_prob')
        if not home_win_prob:
            continue
        
        value_bet = identify_underdog_value(
            game,
            home_win_prob,
            min_ev_threshold=min_ev_threshold
        )
        
        if value_bet:
            # Add game context
            value_bet['game'] = f"{game.get('away_team', 'Away')} @ {game.get('home_team', 'Home')}"
            value_bet['date'] = game.get('game_date', '')
            value_opportunities.append(value_bet)
    
    if not value_opportunities:
        return pd.DataFrame()
    
    df = pd.DataFrame(value_opportunities)
    df = df.sort_values('roi', ascending=False)
    
    return df


def format_value_bet_display(value_bet: Dict) -> str:
    """Format a value bet for display.
    
    Args:
        value_bet: Value bet dictionary
        
    Returns:
        Formatted string for display
    """
    return f"""
    🎯 **VALUE BET DETECTED**
    
    **Team**: {value_bet['team']} (Underdog)
    **Moneyline**: {value_bet['moneyline']:+d}
    
    **Analysis**:
    - Model Probability: {value_bet['model_prob']:.1%}
    - Implied Probability: {value_bet['implied_prob']:.1%}
    - **Edge**: {value_bet['edge']:.1%}
    
    **Expected Value**:
    - EV per $100: ${value_bet['expected_value']:.2f}
    - **ROI**: {value_bet['roi']:.1f}%
    
    💡 *This underdog has a higher chance of winning than the odds suggest!*
    """


def calculate_kelly_criterion(
    win_probability: float,
    moneyline: int,
    kelly_fraction: float = 0.25
) -> float:
    """Calculate optimal bet size using Kelly Criterion.
    
    Args:
        win_probability: Probability of winning (0-1)
        moneyline: American moneyline odds
        kelly_fraction: Fraction of Kelly to use (default 25% for safety)
        
    Returns:
        Recommended bet size as fraction of bankroll
    """
    if moneyline < 0:
        # Favorite
        b = 100 / abs(moneyline)  # Decimal odds - 1
    else:
        # Underdog
        b = moneyline / 100  # Decimal odds - 1
    
    p = win_probability
    q = 1 - p
    
    # Kelly formula: (bp - q) / b
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly for safety
    recommended = max(0, kelly * kelly_fraction)
    
    return recommended


def get_betting_recommendation(
    value_bet: Dict,
    bankroll: float = 1000,
    kelly_fraction: float = 0.25
) -> Dict:
    """Get betting recommendation with Kelly sizing.
    
    Args:
        value_bet: Value bet dictionary
        bankroll: Total bankroll (default $1000)
        kelly_fraction: Fraction of Kelly to use (default 25%)
        
    Returns:
        Recommendation dictionary
    """
    kelly_size = calculate_kelly_criterion(
        value_bet['model_prob'],
        value_bet['moneyline'],
        kelly_fraction
    )
    
    bet_amount = bankroll * kelly_size
    
    if value_bet['moneyline'] < 0:
        potential_profit = bet_amount * (100 / abs(value_bet['moneyline']))
    else:
        potential_profit = bet_amount * (value_bet['moneyline'] / 100)
    
    return {
        'recommended_bet': bet_amount,
        'kelly_percentage': kelly_size * 100,
        'potential_profit': potential_profit,
        'risk_amount': bet_amount,
        'potential_return': bet_amount + potential_profit
    }


if __name__ == "__main__":
    # Example usage
    print("🎯 Underdog Value Bet Analysis")
    print("=" * 60)
    
    # Example game with betting lines
    example_game = {
        'home_team': 'Duke',
        'away_team': 'UNC',
        'home_moneyline': -150,  # Duke is favorite
        'away_moneyline': +130,  # UNC is underdog
    }
    
    # Example: Model predicts UNC has 45% chance to win
    # Implied probability from +130 is ~43.5%
    # This would be a value bet!
    home_win_prob = 0.55  # Duke 55%, UNC 45%
    
    value_bet = identify_underdog_value(example_game, home_win_prob, min_ev_threshold=0)
    
    if value_bet:
        print(format_value_bet_display(value_bet))
        
        recommendation = get_betting_recommendation(value_bet, bankroll=1000)
        print("\n📊 **Betting Recommendation**")
        print(f"Recommended Bet Size: ${recommendation['recommended_bet']:.2f}")
        print(f"Kelly %: {recommendation['kelly_percentage']:.1f}%")
        print(f"Potential Profit: ${recommendation['potential_profit']:.2f}")
    else:
        print("No value bets found in example.")
    
    print("\n" + "=" * 60)
    print("✅ Underdog value analysis module ready!")
