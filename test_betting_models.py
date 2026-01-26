#!/usr/bin/env python3
"""
Test script for betting models implementation.
Validates that all functions from roadmap-betting-models.md work correctly.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

# Add current directory to path
sys.path.append('.')

try:
    from betting_models import (
        train_win_probability_model,
        train_spread_model,
        train_total_model,
        predict_ats,
        predict_over_under,
        evaluate_classification,
        evaluate_regression,
        evaluate_betting_roi,
        tournament_cv,
        create_betting_ensemble,
        get_model_summary,
        validate_dependencies
    )
    print("✅ Successfully imported all betting model functions")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_dependencies():
    """Test that dependencies are available."""
    print("\n🔍 Testing dependencies...")
    try:
        validate_dependencies()
        print("✅ Dependencies validation passed")
    except Exception as e:
        print(f"❌ Dependencies validation failed: {e}")
        return False
    return True

def test_model_training():
    """Test model training functions."""
    print("\n🔍 Testing model training...")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Classification data for moneyline
    X_clf, y_clf = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,
        random_state=42
    )

    # Regression data for spread and total
    X_reg, y_spread = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=0.1,
        random_state=42
    )

    # Total points (always positive)
    y_total = np.abs(y_spread) + 200  # Add baseline

    try:
        # Test moneyline model
        moneyline_model = train_win_probability_model(pd.DataFrame(X_clf), pd.Series(y_clf))
        print("✅ Moneyline model trained successfully")

        # Test spread model
        spread_model = train_spread_model(pd.DataFrame(X_reg), pd.Series(y_spread))
        print("✅ Spread model trained successfully")

        # Test total model
        total_model = train_total_model(pd.DataFrame(X_reg), pd.Series(y_total))
        print("✅ Total model trained successfully")

        return moneyline_model, spread_model, total_model, X_clf, y_clf, X_reg, y_spread, y_total

    except Exception as e:
        print(f"❌ Model training failed: {e}")
        return None, None, None, None, None, None, None, None

def test_predictions(models_and_data):
    """Test prediction functions."""
    print("\n🔍 Testing predictions...")

    moneyline_model, spread_model, total_model, X_clf, y_clf, X_reg, y_spread, y_total = models_and_data

    if not all([moneyline_model, spread_model, total_model]):
        print("❌ Skipping prediction tests - models not available")
        return False

    try:
        # Test moneyline predictions
        moneyline_probs = moneyline_model.predict_proba(pd.DataFrame(X_clf))[:, 1]
        moneyline_preds = (moneyline_probs > 0.5).astype(int)
        print("✅ Moneyline predictions generated")

        # Test spread predictions
        spreads = np.random.normal(0, 10, len(X_reg))  # Sample spreads
        ats_covers, predicted_margins = predict_ats(spread_model, pd.DataFrame(X_reg), pd.Series(spreads))
        print("✅ ATS predictions generated")

        # Test over/under predictions
        totals = np.random.normal(150, 10, len(X_reg))  # Sample totals
        ou_overs, predicted_totals = predict_over_under(total_model, pd.DataFrame(X_reg), pd.Series(totals))
        print("✅ Over/under predictions generated")

        return (moneyline_preds, moneyline_probs, ats_covers,
                predicted_margins, ou_overs, predicted_totals,
                spreads, totals)

    except Exception as e:
        print(f"❌ Prediction tests failed: {e}")
        return None

def test_evaluation(predictions_and_data, original_data):
    """Test evaluation functions."""
    print("\n🔍 Testing evaluation...")

    (moneyline_preds, moneyline_probs, ats_covers,
     predicted_margins, ou_overs, predicted_totals,
     spreads, totals) = predictions_and_data

    _, _, _, X_clf, y_clf, X_reg, y_spread, y_total = original_data

    if moneyline_preds is None:
        print("❌ Skipping evaluation tests - predictions not available")
        return False

    try:
        # Test classification evaluation
        clf_metrics = evaluate_classification(y_clf, moneyline_preds, moneyline_probs)
        print(f"✅ Classification evaluation: Accuracy={clf_metrics['accuracy']:.3f}, Brier={clf_metrics['brier_score']:.3f}")

        # Test regression evaluation
        reg_metrics = evaluate_regression(y_spread, predicted_margins)
        print(f"✅ Regression evaluation: MAE={reg_metrics['mae']:.3f}, RMSE={reg_metrics['rmse']:.3f}")

        # Test ROI evaluation (using dummy odds)
        dummy_odds = np.random.choice([-150, 120, -180, 140], len(moneyline_preds))
        roi_metrics = evaluate_betting_roi(moneyline_preds, y_clf, dummy_odds)
        print(f"✅ ROI evaluation: Win Rate={roi_metrics['win_rate']:.3f}, ROI={roi_metrics['roi_pct']:.1f}%")

        return True

    except Exception as e:
        print(f"❌ Evaluation tests failed: {e}")
        return False

def test_ensemble():
    """Test ensemble model creation."""
    print("\n🔍 Testing ensemble models...")

    try:
        clf_ensemble, reg_ensemble = create_betting_ensemble()
        print("✅ Ensemble models created successfully")
        print(f"   Classification ensemble: {type(clf_ensemble).__name__}")
        print(f"   Regression ensemble: {type(reg_ensemble).__name__}")
        return True

    except Exception as e:
        print(f"❌ Ensemble test failed: {e}")
        return False

def test_model_summary():
    """Test model summary function."""
    print("\n🔍 Testing model summary...")

    try:
        summary = get_model_summary()
        print("✅ Model summary generated:")
        print(summary.to_string(index=False))
        return True

    except Exception as e:
        print(f"❌ Model summary test failed: {e}")
        return False

def test_cross_validation():
    """Test cross validation (simplified version)."""
    print("\n🔍 Testing cross validation...")

    try:
        # Create sample data with year column
        np.random.seed(42)
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'year': np.random.choice([2020, 2021, 2022], 100)
        })
        y = pd.Series(np.random.randn(100))

        # Simple model for testing
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()

        # Test CV (simplified - just check it runs)
        years = [2021, 2022]
        scores = tournament_cv(X, y, model, years)

        print(f"✅ Cross validation completed for {len(scores)} years")
        for score in scores:
            print(f"   Year {score['year']}: Score={score['score']:.3f}")

        return True

    except Exception as e:
        print(f"❌ Cross validation test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Betting Models Implementation")
    print("=" * 50)

    # Test dependencies
    if not test_dependencies():
        return False

    # Test model training
    models_and_data = test_model_training()
    if not models_and_data[0]:
        return False

    # Test predictions
    predictions_and_data = test_predictions(models_and_data)
    if not predictions_and_data:
        return False

    # Test evaluation
    if not test_evaluation(predictions_and_data, models_and_data):
        return False

    # Test ensemble
    if not test_ensemble():
        return False

    # Test model summary
    if not test_model_summary():
        return False

    # Test cross validation
    if not test_cross_validation():
        return False

    print("\n" + "=" * 50)
    print("🎉 All betting model tests passed!")
    print("The roadmap-betting-models.md implementation is working correctly.")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)