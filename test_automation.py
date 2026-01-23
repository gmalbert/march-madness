#!/usr/bin/env python3
"""
Test script for GitHub Actions workflow.
This script can be used to manually test the automated data collection.
"""

import subprocess
import sys
from pathlib import Path

def test_fetch_games():
    """Test the fetch_espn_cbb_scores.py script."""
    print("🧪 Testing automated game fetching...")

    try:
        # Run the fetch script
        result = subprocess.run([sys.executable, "fetch_espn_cbb_scores.py"],
                              capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("✅ Game fetching completed successfully")
            print(f"Output: {result.stdout.strip()}")

            # Check if file was created/updated
            csv_file = Path("data_files/espn_cbb_current_season.csv")
            if csv_file.exists():
                print(f"✅ Data file updated: {csv_file}")
                # Count lines (subtract 1 for header)
                with open(csv_file, 'r') as f:
                    line_count = sum(1 for line in f) - 1
                print(f"📊 Games in dataset: {line_count}")
            else:
                print("❌ Data file not found")

        else:
            print("❌ Game fetching failed")
            print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

def test_generate_predictions():
    """Test the generate_predictions.py script."""
    print("🧪 Testing prediction generation...")

    try:
        # Run the prediction generation script
        result = subprocess.run([sys.executable, "generate_predictions.py"],
                              capture_output=True, text=True, cwd=Path(__file__).parent)

        if result.returncode == 0:
            print("✅ Prediction generation completed successfully")
            print(f"Output: {result.stdout.strip()}")

            # Check if predictions file was created/updated
            predictions_file = Path("data_files/upcoming_game_predictions.json")
            if predictions_file.exists():
                print(f"✅ Predictions file updated: {predictions_file}")

                # Count games with predictions
                with open(predictions_file, 'r') as f:
                    import json
                    predictions_data = json.load(f)
                    game_count = len(predictions_data)
                print(f"📊 Games with predictions: {game_count}")

                if game_count > 0:
                    # Show sample prediction
                    sample_game = predictions_data[0]
                    home_team = sample_game['game_info']['home_team']
                    away_team = sample_game['game_info']['away_team']
                    spread_pred = sample_game['predictions'].get('spread_advanced', {}).get('prediction', sample_game['predictions'].get('spread', {}).get('prediction', 'N/A'))
                    moneyline_pred = sample_game['predictions'].get('moneyline_advanced', {}).get('prediction', sample_game['predictions'].get('moneyline', {}).get('prediction', 'N/A'))
                    print(f"🎯 Sample: {away_team} @ {home_team}")
                    print(f"   Spread: {spread_pred:+.1f} | Moneyline: {moneyline_pred}")

            else:
                print("❌ Predictions file not found")

        else:
            print("❌ Prediction generation failed")
            print(f"Error: {result.stderr}")

    except Exception as e:
        print(f"❌ Test failed with exception: {e}")

if __name__ == "__main__":
    test_fetch_games()
    print()
    test_generate_predictions()