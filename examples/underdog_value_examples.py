"""
Example: Finding Underdog Value Bets

This script demonstrates how to identify value betting opportunities
where underdogs have a higher probability of winning than the odds suggest.
"""

import sys
import os
# Add parent directory to path to import underdog_value
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from underdog_value import (
    identify_underdog_value,
    format_value_bet_display,
    get_betting_recommendation,
    analyze_all_games_for_value,
    moneyline_to_implied_probability,
    calculate_expected_value
)

def example_1_basic_value_detection():
    """Example 1: Basic underdog value detection"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Underdog Value Detection")
    print("="*60)
    
    # Example game: #8 seed vs #1 seed in tournament
    game = {
        'home_team': 'Kansas (1 seed)',
        'away_team': 'NC State (8 seed)',
        'home_moneyline': -350,  # Kansas heavy favorite
        'away_moneyline': +275   # NC State underdog
    }
    
    # Model predicts NC State has 40% chance (better than +275 suggests)
    home_win_prob = 0.60  # Kansas 60%, NC State 40%
    
    print(f"\nGame: {game['away_team']} @ {game['home_team']}")
    print(f"Odds: {game['away_team']} {game['away_moneyline']:+d}")
    print(f"Model: NC State has {(1-home_win_prob):.1%} chance to win")
    
    # Implied probability from +275 is about 26.7%
    implied_prob = moneyline_to_implied_probability(game['away_moneyline'])
    print(f"Implied probability from odds: {implied_prob:.1%}")
    print(f"Model gives NC State {(1-home_win_prob) - implied_prob:.1%} edge!")
    
    value_bet = identify_underdog_value(game, home_win_prob, min_ev_threshold=3.0)
    
    if value_bet:
        print(format_value_bet_display(value_bet))
        
        # Get betting recommendation
        recommendation = get_betting_recommendation(value_bet, bankroll=1000)
        print("\n📊 Betting Recommendation (for $1000 bankroll):")
        print(f"  Bet Size: ${recommendation['recommended_bet']:.2f}")
        print(f"  Kelly %: {recommendation['kelly_percentage']:.1f}%")
        print(f"  Potential Profit: ${recommendation['potential_profit']:.2f}")
    else:
        print("\n❌ No value bet detected")


def example_2_close_game_value():
    """Example 2: Close game with small edge"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Close Game Value")
    print("="*60)
    
    game = {
        'home_team': 'Duke',
        'away_team': 'UNC',
        'home_moneyline': -120,
        'away_moneyline': +100
    }
    
    # Model predicts a coin flip, but UNC is getting plus odds
    home_win_prob = 0.52  # Duke 52%, UNC 48%
    
    print(f"\nGame: {game['away_team']} @ {game['home_team']}")
    print(f"Model: UNC {(1-home_win_prob):.1%}, Duke {home_win_prob:.1%}")
    
    value_bet = identify_underdog_value(game, home_win_prob, min_ev_threshold=2.0)
    
    if value_bet:
        print(f"\n✅ Small edge detected!")
        print(f"  Team: {value_bet['team']}")
        print(f"  Edge: {value_bet['edge']:.1%}")
        print(f"  ROI: {value_bet['roi']:.1f}%")


def example_3_no_value():
    """Example 3: No value bet - odds are accurate"""
    print("\n" + "="*60)
    print("EXAMPLE 3: No Value Bet (Efficient Market)")
    print("="*60)
    
    game = {
        'home_team': 'Villanova',
        'away_team': 'Georgetown',
        'home_moneyline': -200,
        'away_moneyline': +165
    }
    
    # Model agrees with odds - no edge
    home_win_prob = 0.67  # Villanova 67%, Georgetown 33%
    
    print(f"\nGame: {game['away_team']} @ {game['home_team']}")
    
    implied_away = moneyline_to_implied_probability(game['away_moneyline'])
    model_away = 1 - home_win_prob
    
    print(f"Model: Georgetown {model_away:.1%}")
    print(f"Implied: Georgetown {implied_away:.1%}")
    print(f"Edge: {(model_away - implied_away):.1%}")
    
    value_bet = identify_underdog_value(game, home_win_prob, min_ev_threshold=5.0)
    
    if value_bet:
        print("\n✅ Value bet found!")
    else:
        print("\n❌ No significant value - odds are accurate")


def example_4_multiple_games():
    """Example 4: Analyze multiple games for best value"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Analyzing Multiple Games")
    print("="*60)
    
    games = [
        {
            'home_team': 'Kansas', 'away_team': 'Texas Tech',
            'home_moneyline': -180, 'away_moneyline': +150,
            'game_date': '2024-03-21'
        },
        {
            'home_team': 'UConn', 'away_team': 'San Diego State',
            'home_moneyline': -250, 'away_moneyline': +200,
            'game_date': '2024-03-22'
        },
        {
            'home_team': 'Purdue', 'away_team': 'Fairleigh Dickinson',
            'home_moneyline': -2000, 'away_moneyline': +1000,
            'game_date': '2024-03-23'
        }
    ]
    
    # Model predictions (home win probabilities)
    predictions = [
        {'moneyline': {'home_win_prob': 0.62}},  # Texas Tech has value
        {'moneyline': {'home_win_prob': 0.70}},  # San Diego State has value
        {'moneyline': {'home_win_prob': 0.92}},  # Huge upset potential!
    ]
    
    print("\nAnalyzing tournament games for value bets...\n")
    
    value_opportunities = analyze_all_games_for_value(games, predictions, min_ev_threshold=5.0)
    
    if not value_opportunities.empty:
        print(f"Found {len(value_opportunities)} value betting opportunities:\n")
        for idx, row in value_opportunities.iterrows():
            print(f"🎯 {row['game']}")
            print(f"   Bet: {row['team']} ({row['moneyline']:+d})")
            print(f"   Edge: {row['edge']:.1%} | ROI: {row['roi']:.1f}%")
            print(f"   EV per $100: ${row['expected_value']:.2f}\n")
        
        print(f"\n💡 Best opportunity: {value_opportunities.iloc[0]['game']}")
        print(f"   ROI: {value_opportunities.iloc[0]['roi']:.1f}%")
    else:
        print("No value bets found at 5% threshold")


def example_5_ev_calculation():
    """Example 5: Understanding Expected Value"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Expected Value Explained")
    print("="*60)
    
    print("\nScenario: 15-seed upset special")
    print("Model gives 15-seed 20% chance to win")
    print("Moneyline: +1200 (12-to-1 odds)")
    
    win_prob = 0.20
    moneyline = +1200
    
    ev, roi = calculate_expected_value(win_prob, moneyline, bet_amount=100)
    
    print(f"\nMath:")
    print(f"  If win (20% chance): Win $1200")
    print(f"  If lose (80% chance): Lose $100")
    print(f"  Expected Value: (0.20 × $1200) - (0.80 × $100)")
    print(f"  EV = ${ev:.2f}")
    print(f"  ROI = {roi:.1f}%")
    
    if ev > 0:
        print(f"\n✅ This is a +EV bet! Over many trials, expect to win ${ev:.2f} per $100 bet")
    
    # Compare to implied probability
    implied = moneyline_to_implied_probability(moneyline)
    print(f"\nOdds imply {implied:.1%} chance")
    print(f"Model says {win_prob:.1%} chance")
    print(f"Edge: {(win_prob - implied):.1%}")


if __name__ == "__main__":
    print("\n🎯 UNDERDOG VALUE BET EXAMPLES")
    print("=" * 60)
    print("Demonstrating how to find profitable underdog bets")
    
    example_1_basic_value_detection()
    example_2_close_game_value()
    example_3_no_value()
    example_4_multiple_games()
    example_5_ev_calculation()
    
    print("\n" + "="*60)
    print("✅ All examples complete!")
    print("\n💡 Key Takeaways:")
    print("  1. Value exists when model probability > implied probability")
    print("  2. Underdogs with +EV are the best opportunities")
    print("  3. Kelly Criterion helps size bets for long-term growth")
    print("  4. Small edges add up over many bets")
    print("  5. Always bet responsibly within your bankroll")
    print("="*60 + "\n")
