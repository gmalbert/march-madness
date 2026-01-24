"""
Compare model performance with and without KenPom/BartTorvik features.

Trains two sets of models:
1. Baseline: 3 features (off_eff_diff, def_eff_diff, net_eff_diff)
2. Extended: 11 features (3 baseline + 6 KenPom + 2 BartTorvik)

Reports improvement in MAE and accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import json
from pathlib import Path

DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Feature sets
BASELINE_FEATURES = ['off_eff_diff', 'def_eff_diff', 'net_eff_diff']
KENPOM_FEATURES = ['kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff', 
                   'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff']
BART_FEATURES = ['bart_oe_diff', 'bart_de_diff']
EXTENDED_FEATURES = BASELINE_FEATURES + KENPOM_FEATURES + BART_FEATURES


def load_data():
    """Load enriched training data with complete feature set."""
    # Use complete cases where all features are available
    complete_path = DATA_DIR / "training_data_complete_features.csv"
    enriched_path = DATA_DIR / "training_data_enriched.csv"
    
    if complete_path.exists():
        df = pd.read_csv(complete_path)
        print(f"✅ Loaded {len(df)} games with complete features")
    elif enriched_path.exists():
        df = pd.read_csv(enriched_path)
        # Filter to complete cases
        df = df.dropna(subset=EXTENDED_FEATURES)
        print(f"✅ Loaded {len(df)} games with complete features from enriched data")
    else:
        print("❌ No enriched training data found. Run enrich_training_data.py first.")
        return None
    
    print(f"   Years: {sorted(df['season'].unique())}")
    print(f"   Regular games: {len(df[df['game_type']=='regular'])}")
    print(f"   Tournament games: {len(df[df['game_type']=='tournament'])}")
    
    return df


def train_spread_models(df, feature_set, label="Baseline"):
    """Train spread prediction models with specified feature set."""
    print(f"\n{'='*60}")
    print(f"{label} SPREAD MODELS ({len(feature_set)} features)")
    print(f"{'='*60}")
    
    # Prepare data
    valid_games = df.dropna(subset=['actual_spread'])
    X = valid_games[feature_set]
    y = valid_games['actual_spread']
    weights = valid_games['sample_weight']
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train, sample_weight=w_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    results['linear_regression'] = {'mae': lr_mae, 'model': lr, 'scaler': scaler}
    print(f"  Linear Regression MAE: {lr_mae:.2f}")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=w_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    results['random_forest'] = {'mae': rf_mae, 'model': rf}
    print(f"  Random Forest MAE: {rf_mae:.2f}")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    results['xgboost'] = {'mae': xgb_mae, 'model': xgb_model}
    print(f"  XGBoost MAE: {xgb_mae:.2f}")
    
    # Ensemble average
    ensemble_mae = np.mean([lr_mae, rf_mae, xgb_mae])
    results['ensemble_mae'] = ensemble_mae
    print(f"  Ensemble MAE: {ensemble_mae:.2f}")
    
    return results


def train_total_models(df, feature_set, label="Baseline"):
    """Train total prediction models with specified feature set."""
    print(f"\n{'='*60}")
    print(f"{label} TOTAL MODELS ({len(feature_set)} features)")
    print(f"{'='*60}")
    
    # Prepare data
    valid_games = df.dropna(subset=['actual_total'])
    X = valid_games[feature_set]
    y = valid_games['actual_total']
    weights = valid_games['sample_weight']
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # Linear Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train, sample_weight=w_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    results['linear_regression'] = {'mae': lr_mae, 'model': lr, 'scaler': scaler}
    print(f"  Linear Regression MAE: {lr_mae:.2f}")
    
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=w_train)
    rf_pred = rf.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    results['random_forest'] = {'mae': rf_mae, 'model': rf}
    print(f"  Random Forest MAE: {rf_mae:.2f}")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    results['xgboost'] = {'mae': xgb_mae, 'model': xgb_model}
    print(f"  XGBoost MAE: {xgb_mae:.2f}")
    
    # Ensemble average
    ensemble_mae = np.mean([lr_mae, rf_mae, xgb_mae])
    results['ensemble_mae'] = ensemble_mae
    print(f"  Ensemble MAE: {ensemble_mae:.2f}")
    
    return results


def train_moneyline_models(df, feature_set, label="Baseline"):
    """Train moneyline prediction models with specified feature set."""
    print(f"\n{'='*60}")
    print(f"{label} MONEYLINE MODELS ({len(feature_set)} features)")
    print(f"{'='*60}")
    
    # Prepare data
    valid_games = df.dropna(subset=['actual_spread'])
    X = valid_games[feature_set]
    y = (valid_games['actual_spread'] < 0).astype(int)  # 1 if home won
    weights = valid_games['sample_weight']
    
    # Split data
    X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
        X, y, weights, test_size=0.2, random_state=42, stratify=y
    )
    
    results = {}
    
    # Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train, y_train, sample_weight=w_train)
    lr_pred = lr.predict(X_test)
    lr_acc = accuracy_score(y_test, lr_pred)
    results['logistic_regression'] = {'accuracy': lr_acc, 'model': lr}
    print(f"  Logistic Regression Accuracy: {lr_acc:.1%}")
    
    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train, sample_weight=w_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results['random_forest'] = {'accuracy': rf_acc, 'model': rf}
    print(f"  Random Forest Accuracy: {rf_acc:.1%}")
    
    # XGBoost
    xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
    xgb_model.fit(X_train, y_train, sample_weight=w_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    results['xgboost'] = {'accuracy': xgb_acc, 'model': xgb_model}
    print(f"  XGBoost Accuracy: {xgb_acc:.1%}")
    
    # Ensemble average
    ensemble_acc = np.mean([lr_acc, rf_acc, xgb_acc])
    results['ensemble_accuracy'] = ensemble_acc
    print(f"  Ensemble Accuracy: {ensemble_acc:.1%}")
    
    return results


def save_extended_models(spread_results, total_results, moneyline_results):
    """Save the extended feature models."""
    print("\n💾 Saving extended feature models...")
    
    # Save spread models
    for name, data in spread_results.items():
        if name != 'ensemble_mae' and 'model' in data:
            path = MODEL_DIR / f"spread_{name}.joblib"
            joblib.dump(data['model'], path)
            if 'scaler' in data:
                scaler_path = MODEL_DIR / f"spread_{name}_scaler.joblib"
                joblib.dump(data['scaler'], scaler_path)
    
    # Save total models
    for name, data in total_results.items():
        if name != 'ensemble_mae' and 'model' in data:
            path = MODEL_DIR / f"total_{name}.joblib"
            joblib.dump(data['model'], path)
            if 'scaler' in data:
                scaler_path = MODEL_DIR / f"total_{name}_scaler.joblib"
                joblib.dump(data['scaler'], scaler_path)
    
    # Save moneyline models
    for name, data in moneyline_results.items():
        if name != 'ensemble_accuracy' and 'model' in data:
            path = MODEL_DIR / f"moneyline_{name}.joblib"
            joblib.dump(data['model'], path)
    
    # Save metrics
    spread_mae_values = [r['mae'] for r in spread_results.values() if isinstance(r, dict) and 'mae' in r]
    spread_metrics = {
        'mae': spread_results['ensemble_mae'],
        'mae_range': f"{min(spread_mae_values):.2f} - {max(spread_mae_values):.2f}"
    }
    with open(MODEL_DIR / "spread_metrics.json", 'w') as f:
        json.dump(spread_metrics, f, indent=2)
    
    total_mae_values = [r['mae'] for r in total_results.values() if isinstance(r, dict) and 'mae' in r]
    total_metrics = {
        'mae': total_results['ensemble_mae'],
        'mae_range': f"{min(total_mae_values):.2f} - {max(total_mae_values):.2f}"
    }
    with open(MODEL_DIR / "total_metrics.json", 'w') as f:
        json.dump(total_metrics, f, indent=2)
    
    moneyline_acc_values = [r['accuracy'] for r in moneyline_results.values() if isinstance(r, dict) and 'accuracy' in r]
    moneyline_metrics = {
        'accuracy': moneyline_results['ensemble_accuracy'],
        'accuracy_range': f"{min(moneyline_acc_values):.1%} - {max(moneyline_acc_values):.1%}"
    }
    with open(MODEL_DIR / "moneyline_metrics.json", 'w') as f:
        json.dump(moneyline_metrics, f, indent=2)
    
    print(f"✅ Models saved to {MODEL_DIR}")


def main():
    print("🏀 Model Performance Comparison: Baseline vs Extended Features\n")
    
    # Load data
    df = load_data()
    if df is None:
        return
    
    print(f"\n📊 Training on {len(df)} games with complete feature coverage")
    
    # Train baseline models (3 features)
    print("\n" + "="*70)
    print("BASELINE MODELS (CBBD Features Only)")
    print("="*70)
    baseline_spread = train_spread_models(df, BASELINE_FEATURES, "Baseline")
    baseline_total = train_total_models(df, BASELINE_FEATURES, "Baseline")
    baseline_moneyline = train_moneyline_models(df, BASELINE_FEATURES, "Baseline")
    
    # Train extended models (11 features)
    print("\n" + "="*70)
    print("EXTENDED MODELS (CBBD + KenPom + BartTorvik)")
    print("="*70)
    extended_spread = train_spread_models(df, EXTENDED_FEATURES, "Extended")
    extended_total = train_total_models(df, EXTENDED_FEATURES, "Extended")
    extended_moneyline = train_moneyline_models(df, EXTENDED_FEATURES, "Extended")
    
    # Calculate improvements
    print("\n" + "="*70)
    print("📈 PERFORMANCE IMPROVEMENT")
    print("="*70)
    
    spread_improvement = baseline_spread['ensemble_mae'] - extended_spread['ensemble_mae']
    spread_pct = (spread_improvement / baseline_spread['ensemble_mae']) * 100
    print(f"\n🎯 SPREAD:")
    print(f"   Baseline MAE:  {baseline_spread['ensemble_mae']:.2f} points")
    print(f"   Extended MAE:  {extended_spread['ensemble_mae']:.2f} points")
    print(f"   Improvement:   {spread_improvement:.2f} points ({spread_pct:+.1f}%)")
    
    total_improvement = baseline_total['ensemble_mae'] - extended_total['ensemble_mae']
    total_pct = (total_improvement / baseline_total['ensemble_mae']) * 100
    print(f"\n🎯 TOTAL:")
    print(f"   Baseline MAE:  {baseline_total['ensemble_mae']:.2f} points")
    print(f"   Extended MAE:  {extended_total['ensemble_mae']:.2f} points")
    print(f"   Improvement:   {total_improvement:.2f} points ({total_pct:+.1f}%)")
    
    moneyline_improvement = extended_moneyline['ensemble_accuracy'] - baseline_moneyline['ensemble_accuracy']
    moneyline_pct = moneyline_improvement * 100
    print(f"\n🎯 MONEYLINE:")
    print(f"   Baseline Acc:  {baseline_moneyline['ensemble_accuracy']:.1%}")
    print(f"   Extended Acc:  {extended_moneyline['ensemble_accuracy']:.1%}")
    print(f"   Improvement:   {moneyline_pct:+.1f} percentage points")
    
    # Save extended models
    save_extended_models(extended_spread, extended_total, extended_moneyline)
    
    # Create comparison report
    report = {
        'spread': {
            'baseline_mae': baseline_spread['ensemble_mae'],
            'extended_mae': extended_spread['ensemble_mae'],
            'improvement_points': spread_improvement,
            'improvement_percent': spread_pct
        },
        'total': {
            'baseline_mae': baseline_total['ensemble_mae'],
            'extended_mae': extended_total['ensemble_mae'],
            'improvement_points': total_improvement,
            'improvement_percent': total_pct
        },
        'moneyline': {
            'baseline_accuracy': baseline_moneyline['ensemble_accuracy'],
            'extended_accuracy': extended_moneyline['ensemble_accuracy'],
            'improvement_percentage_points': moneyline_pct
        }
    }
    
    with open(MODEL_DIR / "model_comparison.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Comparison report saved to {MODEL_DIR / 'model_comparison.json'}")
    print("\n🎉 Model retraining complete!")
    print("   Extended models with KenPom + BartTorvik features are now active.")


if __name__ == '__main__':
    main()
