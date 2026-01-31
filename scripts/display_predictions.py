#!/usr/bin/env python3
"""
Display upcoming game predictions in a readable format.
"""

import json
from pathlib import Path
from datetime import datetime

def display_predictions():
    """Display predictions for upcoming games."""
    predictions_file = Path("data_files/upcoming_game_predictions.json")

    if not predictions_file.exists():
        print("No predictions file found. Run generate_predictions.py first.")
        return

    with open(predictions_file, 'r') as f:
        predictions_data = json.load(f)

    print("🏀 Upcoming College Basketball Game Predictions")
    print("=" * 60)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Games with predictions: {len(predictions_data)}")
    print()

    for game in predictions_data:
        game_info = game['game_info']
        predictions = game['predictions']

        # Game header
        home_team = game_info['home_team']
        away_team = game_info['away_team']
        date_str = game_info.get('date', 'TBD')

        try:
            game_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            date_display = game_date.strftime('%a %b %d, %I:%M %p ET')
        except:
            date_display = date_str

        print(f"🎯 {away_team} @ {home_team}")
        print(f"   📅 {date_display}")
        if game_info.get('venue'):
            print(f"   📍 {game_info['venue']}")

        # Predictions
        if 'moneyline_home_win_prob' in predictions:
            home_prob = predictions['moneyline_home_win_prob']
            away_prob = predictions['moneyline_away_win_prob']
            prediction = "Home" if home_prob > 0.5 else "Away"
            confidence = f"{max(home_prob, away_prob):.1%}"
            print(f"   💰 Moneyline: {prediction} ({confidence})")

        if 'spread_prediction' in predictions:
            spread = predictions['spread_prediction']
            print(f"   📏 Spread: {spread:+.1f} points")

        if 'total_prediction' in predictions:
            total = predictions['total_prediction']
            print(f"   🔢 Total: {total:.1f} points")

        # Basic model predictions (if advanced not available)
        if 'moneyline' in predictions and 'moneyline_advanced' not in predictions:
            ml = predictions['moneyline']
            print(f"   💰 Moneyline: {ml['prediction']} ({ml['confidence']})")

        if 'spread' in predictions and 'spread_advanced' not in predictions:
            spread = predictions['spread']
            print(f"   📏 Spread: {spread['prediction']:+.1f} points")

        if 'total' in predictions and 'total_advanced' not in predictions:
            total = predictions['total']
            print(f"   🔢 Total: {total['prediction']:.1f} points")

        print()

if __name__ == "__main__":
    display_predictions()