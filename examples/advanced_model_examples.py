"""
Examples: Advanced Betting Model Training

Demonstrates the roadmap betting models with calibrated classifiers,
time-based cross validation, and ensemble approaches.
"""

import sys
import os
# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from advanced_model_training import (
    load_weighted_training_data,
    prepare_features,
    train_win_probability_model,
    train_spread_model,
    train_total_model,
    predict_ats,
    predict_over_under,
    evaluate_classification,
    evaluate_regression,
    evaluate_betting_roi,
    tournament_cv,
    create_betting_ensemble
)
import xgboost as xgb

def example_1_calibrated_moneyline():
    """Example 1: Calibrated moneyline model for better probability estimates."""
    print("\n" + "="*70)
    print("EXAMPLE 1: CALIBRATED MONEYLINE MODEL")
    print("="*70)

    # Load data
    df = load_weighted_training_data()
    if df.empty:
        return

    # Prepare features
    X, y, weights = prepare_features(df, 'moneyline')

    # Train calibrated model
    model = train_win_probability_model(X, y, weights)

    # Demonstrate calibration quality
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Get calibrated probabilities
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Evaluate calibration
    metrics = evaluate_classification(y_test, y_pred, y_prob)

    print("\n📊 Calibration Results:")
    print(f"  Accuracy: {metrics['accuracy']:.1%}")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Log Loss: {metrics['log_loss']:.4f}")

    # Show probability distribution
    print("\n🎲 Probability Distribution (first 10 predictions):")
    for i in range(min(10, len(y_prob))):
        actual = "WIN" if y_test.iloc[i] == 1 else "LOSS"
        prob = y_prob[i]
        print(f"  {actual}: {prob:.1%}")

    print("\n💡 Calibrated probabilities are essential for converting to betting odds!")


def example_2_spread_prediction():
    """Example 2: Spread prediction with ATS analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 2: SPREAD PREDICTION & ATS ANALYSIS")
    print("="*70)

    df = load_weighted_training_data()
    if df.empty:
        return

    X, y, weights = prepare_features(df, 'spread')
    model = train_spread_model(X, y, weights)

    # Simulate betting scenario
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict margins
    predicted_margins = model.predict(X_test)

    # Simulate betting lines (assume average spread of 8 points)
    betting_spreads = np.full(len(y_test), 8.0)  # Home team favored by 8

    # Predict ATS outcomes
    covers, margins = predict_ats(model, X_test, betting_spreads)

    print("\n🏈 ATS Prediction Results:")
    print(f"Games analyzed: {len(covers)}")
    print(f"Predicted covers: {covers.sum()}")
    print(f"Predicted no-covers: {(~covers).sum()}")

    # Show sample predictions
    print("\n📋 Sample Predictions:")
    for i in range(min(5, len(predicted_margins))):
        pred_margin = predicted_margins[i]
        bet_spread = betting_spreads[i]
        actual_margin = y_test.iloc[i]
        covers_pred = "COVER" if covers[i] else "NO COVER"
        covers_actual = "COVER" if actual_margin > bet_spread else "NO COVER"

        print(f"Game {i+1}: Pred {pred_margin:+.1f} vs Line {bet_spread:+.1f} | {covers_pred} (Actual: {covers_actual})")


def example_3_over_under_prediction():
    """Example 3: Total prediction with over/under analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 3: TOTAL PREDICTION & OVER/UNDER ANALYSIS")
    print("="*70)

    df = load_weighted_training_data()
    if df.empty:
        return

    X, y, weights = prepare_features(df, 'total')
    model = train_total_model(X, y, weights)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Predict totals
    predicted_totals = model.predict(X_test)

    # Simulate betting lines (assume average total of 145)
    betting_totals = np.full(len(y_test), 145.0)

    # Predict over/under
    overs, totals = predict_over_under(model, X_test, betting_totals)

    print("\n🔢 Over/Under Prediction Results:")
    print(f"Games analyzed: {len(overs)}")
    print(f"Predicted overs: {overs.sum()}")
    print(f"Predicted unders: {(~overs).sum()}")

    # Show sample predictions
    print("\n📋 Sample Predictions:")
    for i in range(min(5, len(predicted_totals))):
        pred_total = predicted_totals[i]
        bet_total = betting_totals[i]
        actual_total = y_test.iloc[i]
        ou_pred = "OVER" if overs[i] else "UNDER"
        ou_actual = "OVER" if actual_total > bet_total else "UNDER"

        print(f"Game {i+1}: Pred {pred_total:.1f} vs Line {bet_total:.1f} | {ou_pred} (Actual: {ou_actual})")


def example_4_ensemble_models():
    """Example 4: Ensemble model comparison."""
    print("\n" + "="*70)
    print("EXAMPLE 4: ENSEMBLE MODEL COMPARISON")
    print("="*70)

    df = load_weighted_training_data()
    if df.empty:
        return

    # Create ensemble
    clf_ensemble, reg_ensemble = create_betting_ensemble()

    # Test classification ensemble
    X_ml, y_ml, _ = prepare_features(df, 'moneyline')
    if not X_ml.empty:
        from sklearn.model_selection import cross_val_score

        scores = cross_val_score(clf_ensemble, X_ml, y_ml, cv=5, scoring='accuracy')
        print("\n🤖 Classification Ensemble (Moneyline):")
        print(".1%")
        print(".3f")

    # Test regression ensemble
    X_spread, y_spread, _ = prepare_features(df, 'spread')
    if not X_spread.empty:
        scores = cross_val_score(reg_ensemble, X_spread, y_spread, cv=5, scoring='neg_mean_absolute_error')
        print("\n🤖 Regression Ensemble (Spread):")
        print(".2f")
        print(".3f")


def example_5_time_based_cv():
    """Example 5: Time-based cross validation."""
    print("\n" + "="*70)
    print("EXAMPLE 5: TIME-BASED CROSS VALIDATION")
    print("="*70)

    df = load_weighted_training_data()
    if df.empty:
        return

    # Add season column if not present
    if 'season' not in df.columns:
        # Assume season is the year from some date column
        df['season'] = 2024  # Default for demo

    years = sorted(df['season'].unique())
    print(f"Available seasons: {years}")

    # Moneyline time-based CV
    X_ml, y_ml, _ = prepare_features(df, 'moneyline')
    if not X_ml.empty and 'season' in X_ml.columns:
        ml_model = xgb.XGBClassifier(n_estimators=50, random_state=42)
        cv_scores = tournament_cv(X_ml, y_ml, ml_model, years[:3])  # Test first 3 years

        print("\n⏰ Moneyline Time-Based CV Results:")
        for score in cv_scores:
            print(".3f")

    # Spread time-based CV
    X_spread, y_spread, _ = prepare_features(df, 'spread')
    if not X_spread.empty and 'season' in X_spread.columns:
        spread_model = xgb.XGBRegressor(n_estimators=50, random_state=42)
        cv_scores = tournament_cv(X_spread, y_spread, spread_model, years[:3])

        print("\n⏰ Spread Time-Based CV Results:")
        for score in cv_scores:
            print(".3f")


def example_6_betting_roi_analysis():
    """Example 6: Betting ROI analysis."""
    print("\n" + "="*70)
    print("EXAMPLE 6: BETTING ROI ANALYSIS")
    print("="*70)

    # Simulate betting results
    np.random.seed(42)

    # Simulate 100 bets
    n_bets = 100
    predictions = np.random.choice([True, False], n_bets, p=[0.55, 0.45])  # 55% accuracy
    actuals = np.random.choice([True, False], n_bets)  # Random outcomes
    odds = np.random.choice([-150, -120, +120, +150], n_bets)  # Mix of odds

    roi_results = evaluate_betting_roi(predictions, actuals, odds, stake=100)

    print("\n💰 Simulated Betting Results (100 bets at $100 each):")
    print(f"Wins: {roi_results['wins']}")
    print(f"Losses: {roi_results['losses']}")
    print(f"Total Profit: ${roi_results['profit']:.2f}")
    print(".1%")
    print(".1%")

    # Show break-even analysis
    win_rate_needed = 100 / (100 + 120)  # For -120 odds
    print("\n📊 Break-Even Analysis:")
    print(".1%")
    print(".1%")

    if roi_results['win_rate'] > win_rate_needed:
        print("✅ Profitable edge detected!")
    else:
        print("❌ No profitable edge")


if __name__ == "__main__":
    print("\n🎯 ADVANCED BETTING MODEL EXAMPLES")
    print("=" * 70)
    print("Demonstrating roadmap implementations:")
    print("• Calibrated classifiers for moneyline")
    print("• Regression models for spread/total")
    print("• Ensemble approaches")
    print("• Time-based cross validation")
    print("• Comprehensive evaluation metrics")
    print("=" * 70)

    try:
        example_1_calibrated_moneyline()
        example_2_spread_prediction()
        example_3_over_under_prediction()
        example_4_ensemble_models()
        example_5_time_based_cv()
        example_6_betting_roi_analysis()

        print("\n" + "="*70)
        print("✅ All advanced model examples completed!")
        print("\n🚀 Ready to train production models with:")
        print("   python advanced_model_training.py")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have run feature_engineering.py first to create training data.")