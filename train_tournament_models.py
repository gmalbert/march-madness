#!/usr/bin/env python3
"""Train tournament-specific prediction models.

Strategy
--------
Regular-season models see ~25k games; tournament games are only ~1,119 games
across 10 years.  Training dedicated tournament models requires a two-stage
approach:

1. **Warm-start**: fit base models on all data (regular + tournament).
2. **Fine-tune**: retrain with tournament games weighted 3× to push the model
   towards tournament behavior.
3. **Leave-one-year-out CV**: evaluate on each tournament year held out.
4. **Save** tournament models to `data_files/models/tournament_*.joblib`.
5. **Compare** tournament models vs regular-season models on tournament hold-out
   data and report results.

Usage
-----
    python train_tournament_models.py
    python train_tournament_models.py --feature-set enriched
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, brier_score_loss,
                              mean_absolute_error, mean_squared_error)
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_data(feature_set: str = "weighted") -> pd.DataFrame:
    candidates = {
        "enriched": DATA_DIR / "training_data_enriched.csv",
        "weighted": DATA_DIR / "training_data_weighted.csv",
        "comprehensive": DATA_DIR / "training_data_comprehensive.csv",
    }
    path = candidates.get(feature_set, candidates["weighted"])
    if not path.exists():
        # Fallback
        for fallback in candidates.values():
            if fallback.exists():
                path = fallback
                break
    print(f"Loading data from {path}")
    df = pd.read_csv(path)
    print(f"  Rows: {len(df)}  |  Cols: {len(df.columns)}")
    print(f"  Game types: {df['game_type'].value_counts().to_dict()}")
    return df


def get_feature_columns(df: pd.DataFrame, model_type: str) -> List[str]:
    """Return the feature columns for a given model type."""
    if model_type in ("spread", "moneyline"):
        cols = [c for c in df.columns if c.startswith("spread_")]
    elif model_type == "total":
        cols = [c for c in df.columns if c.startswith("total_")]
    else:
        cols = [c for c in df.columns if c.startswith("spread_") or c.startswith("total_")]

    # Add enriched features if present
    enriched = [c for c in df.columns if c.startswith("kenpom_") or c.startswith("bart_")]
    cols += enriched

    return cols


def prepare_xy(df: pd.DataFrame, model_type: str) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, y, weights) for a model type."""
    if model_type in ("spread",):
        target = "actual_spread"
    elif model_type == "total":
        target = "actual_total"
    else:  # moneyline
        target = "actual_spread"   # derive y from sign

    valid = df.dropna(subset=[target]).copy()
    feat_cols = get_feature_columns(valid, model_type)
    feat_cols = [c for c in feat_cols if c in valid.columns]

    X = valid[feat_cols].fillna(0)
    if model_type == "moneyline":
        y = (valid[target] < 0).astype(int)   # 1 = home wins
    else:
        y = valid[target]

    weights = valid.get("sample_weight", pd.Series(np.ones(len(valid)), index=valid.index))
    return X, y, weights


# -----------------------------------------------------------------------------
# Training helpers
# -----------------------------------------------------------------------------

def _tournament_weights(df_valid: pd.DataFrame, base_weight: pd.Series,
                         tourney_mult: float = 3.0) -> np.ndarray:
    """Scale tournament game weights by tourney_mult."""
    w = base_weight.values.copy().astype(float)
    is_tourney = df_valid.get("game_type", pd.Series("regular", index=df_valid.index)) == "tournament"
    w[is_tourney.values] *= tourney_mult
    return w


def train_spread_tournament(X_all: pd.DataFrame, y_all: pd.Series,
                             weights_all: pd.Series) -> Dict:
    """Train tournament-weighted spread models."""
    from sklearn.model_selection import train_test_split

    idx = X_all.index
    # Recreate 80/20 split deterministically
    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
        X_all, y_all, weights_all, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models: Dict[str, Dict] = {}

    # Ridge
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr, sample_weight=w_tr)
    pred = ridge.predict(X_te_s)
    models["ridge"] = dict(model=ridge, scaler=scaler,
                            mae=mean_absolute_error(y_te, pred),
                            rmse=float(np.sqrt(mean_squared_error(y_te, pred))))

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=3, random_state=42)
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = xgb.predict(X_te)
    models["xgboost"] = dict(model=xgb, scaler=None,
                               mae=mean_absolute_error(y_te, pred),
                               rmse=float(np.sqrt(mean_squared_error(y_te, pred))))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                min_samples_split=10, random_state=42)
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = rf.predict(X_te)
    models["random_forest"] = dict(model=rf, scaler=None,
                                    mae=mean_absolute_error(y_te, pred),
                                    rmse=float(np.sqrt(mean_squared_error(y_te, pred))))

    return models


def train_total_tournament(X_all: pd.DataFrame, y_all: pd.Series,
                            weights_all: pd.Series) -> Dict:
    """Train tournament-weighted total models."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
        X_all, y_all, weights_all, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models: Dict[str, Dict] = {}

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr_s, y_tr, sample_weight=w_tr)
    pred = ridge.predict(X_te_s)
    models["ridge"] = dict(model=ridge, scaler=scaler,
                            mae=mean_absolute_error(y_te, pred),
                            rmse=float(np.sqrt(mean_squared_error(y_te, pred))))

    xgb = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        min_child_weight=3, random_state=42)
    xgb.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = xgb.predict(X_te)
    models["xgboost"] = dict(model=xgb, scaler=None,
                               mae=mean_absolute_error(y_te, pred),
                               rmse=float(np.sqrt(mean_squared_error(y_te, pred))))

    rf = RandomForestRegressor(n_estimators=200, max_depth=6,
                                min_samples_split=10, random_state=42)
    rf.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = rf.predict(X_te)
    models["random_forest"] = dict(model=rf, scaler=None,
                                    mae=mean_absolute_error(y_te, pred),
                                    rmse=float(np.sqrt(mean_squared_error(y_te, pred))))

    return models


def train_moneyline_tournament(X_all: pd.DataFrame, y_all: pd.Series,
                                weights_all: pd.Series) -> Dict:
    """Train tournament-weighted moneyline (classification) models."""
    from sklearn.model_selection import train_test_split

    X_tr, X_te, y_tr, y_te, w_tr, _ = train_test_split(
        X_all, y_all, weights_all, test_size=0.2, random_state=42, stratify=y_all)

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    models: Dict[str, Dict] = {}

    # Logistic Regression + calibration
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr_cal = CalibratedClassifierCV(lr, method="isotonic", cv=5)
    lr_cal.fit(X_tr_s, y_tr, sample_weight=w_tr)
    pred = lr_cal.predict(X_te_s)
    prob = lr_cal.predict_proba(X_te_s)[:, 1]
    models["logistic_regression"] = dict(model=lr_cal, scaler=scaler,
                                          accuracy=accuracy_score(y_te, pred),
                                          brier=brier_score_loss(y_te, prob))

    # XGBoost – direct (XGBClassifier already outputs calibrated probabilities)
    xgb_m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8,
                           min_child_weight=3, eval_metric="logloss",
                           random_state=42)
    xgb_m.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = xgb_m.predict(X_te)
    prob = xgb_m.predict_proba(X_te)[:, 1]
    models["xgboost"] = dict(model=xgb_m, scaler=None,
                              accuracy=accuracy_score(y_te, pred),
                              brier=brier_score_loss(y_te, prob))

    # Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                     learning_rate=0.05,
                                     min_samples_split=20, random_state=42)
    gb.fit(X_tr, y_tr, sample_weight=w_tr)
    pred = gb.predict(X_te)
    prob = gb.predict_proba(X_te)[:, 1]
    models["gradient_boosting"] = dict(model=gb, scaler=None,
                                        accuracy=accuracy_score(y_te, pred),
                                        brier=brier_score_loss(y_te, prob))

    return models


# -----------------------------------------------------------------------------
# Leave-one-year-out evaluation on tournament games
# -----------------------------------------------------------------------------

def lovo_cv_tournament(df: pd.DataFrame, model_type: str) -> pd.DataFrame:
    """Leave-one-year-out cross-validation on tournament games.

    Returns a DataFrame with per-year results.
    """
    print(f"\n{'-'*60}")
    print(f"  Leave-one-year-out CV  |  model: {model_type}")
    print(f"{'-'*60}")

    df_tourney = df[df["game_type"] == "tournament"].copy()
    years = sorted(df_tourney["season"].unique())
    feat_cols = get_feature_columns(df, model_type)
    feat_cols = [c for c in feat_cols if c in df.columns]

    records = []
    for year in years:
        test_idx = (df["season"] == year) & (df["game_type"] == "tournament")
        train_idx = ~test_idx

        df_train = df[train_idx].dropna(
            subset=["actual_spread" if model_type != "total" else "actual_total"])
        df_test = df[test_idx].dropna(
            subset=["actual_spread" if model_type != "total" else "actual_total"])

        if len(df_test) < 5:
            continue

        X_tr = df_train[feat_cols].fillna(0)
        X_te = df_test[feat_cols].fillna(0)

        if model_type == "moneyline":
            y_tr = (df_train["actual_spread"] < 0).astype(int)
            y_te = (df_test["actual_spread"] < 0).astype(int)
        elif model_type == "spread":
            y_tr = df_train["actual_spread"]
            y_te = df_test["actual_spread"]
        else:
            y_tr = df_train["actual_total"]
            y_te = df_test["actual_total"]

        w_tr = df_train.get("sample_weight",
                             pd.Series(np.ones(len(df_train)), index=df_train.index))
        # Apply 3× tournament weighting
        is_t = df_train["game_type"] == "tournament"
        w_arr = w_tr.values.copy().astype(float)
        w_arr[is_t.values] *= 3.0

        if model_type == "moneyline":
            model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                   eval_metric="logloss",
                                   random_state=42)
            model.fit(X_tr, y_tr, sample_weight=w_arr)
            preds = model.predict(X_te)
            rec = {"year": year, "n_games": len(y_te),
                   "accuracy": round(accuracy_score(y_te, preds), 4)}
        else:
            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=42)
            model.fit(X_tr, y_tr, sample_weight=w_arr)
            preds = model.predict(X_te)
            rec = {"year": year, "n_games": len(y_te),
                   "mae": round(mean_absolute_error(y_te, preds), 3),
                   "rmse": round(float(np.sqrt(mean_squared_error(y_te, preds))), 3)}

        records.append(rec)
        print(f"  {year}: {rec}")

    return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Save and report
# -----------------------------------------------------------------------------

def save_tournament_models(models: Dict, model_type: str):
    saved = []
    for name, data in models.items():
        path = MODEL_DIR / f"tournament_{model_type}_{name}.joblib"
        joblib.dump(data["model"], path)
        saved.append(str(path))
        if data.get("scaler") is not None:
            sp = MODEL_DIR / f"tournament_{model_type}_{name}_scaler.joblib"
            joblib.dump(data["scaler"], sp)
    print(f"  Saved {len(saved)} {model_type} models -> {MODEL_DIR}")
    return saved


def compare_with_baseline(df: pd.DataFrame, tournament_models: Dict,
                           model_type: str) -> Dict:
    """Compare tournament models vs existing baseline models on tournament test set."""
    df_tourney = df[df["game_type"] == "tournament"].copy()
    feat_cols = get_feature_columns(df, model_type)
    feat_cols = [c for c in feat_cols if c in df_tourney.columns]

    if model_type == "moneyline":
        target_col, is_clf = "actual_spread", True
        df_tourney = df_tourney.dropna(subset=["actual_spread"])
        y = (df_tourney["actual_spread"] < 0).astype(int)
    elif model_type == "spread":
        target_col, is_clf = "actual_spread", False
        df_tourney = df_tourney.dropna(subset=["actual_spread"])
        y = df_tourney["actual_spread"]
    else:
        target_col, is_clf = "actual_total", False
        df_tourney = df_tourney.dropna(subset=["actual_total"])
        y = df_tourney["actual_total"]

    X = df_tourney[feat_cols].fillna(0)

    comparison: Dict[str, Dict] = {}
    for name, data in tournament_models.items():
        m = data["model"]
        sc = data.get("scaler")
        X_input = sc.transform(X) if sc is not None else X.values
        if is_clf:
            preds = m.predict(X_input)
            comparison[f"tournament_{name}"] = {
                "accuracy": round(accuracy_score(y, preds), 4),
                "n_games": len(y),
            }
        else:
            preds = m.predict(X_input)
            comparison[f"tournament_{name}"] = {
                "mae": round(mean_absolute_error(y, preds), 3),
                "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 3),
                "n_games": len(y),
            }

    # Compare vs baseline (saved regular model)
    base_name = "moneyline_xgboost" if is_clf else f"{model_type}_xgboost"
    base_path = MODEL_DIR / f"{base_name}.joblib"
    if base_path.exists():
        try:
            base_model = joblib.load(base_path)
            import sklearn
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if is_clf:
                    preds = base_model.predict(X.values)
                    comparison["baseline_xgboost"] = {
                        "accuracy": round(accuracy_score(y, preds), 4),
                        "n_games": len(y),
                    }
                else:
                    preds = base_model.predict(X.values)
                    comparison["baseline_xgboost"] = {
                        "mae": round(mean_absolute_error(y, preds), 3),
                        "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 3),
                        "n_games": len(y),
                    }
        except Exception as e:
            print(f"  Could not load baseline model {base_path}: {e}")

    return comparison


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(feature_set: str = "enriched"):
    print("\n" + "=" * 65)
    print("   TOURNAMENT MODEL TRAINING  (train_tournament_models.py)")
    print("=" * 65)

    df = load_data(feature_set)

    all_results: Dict = {}

    for model_type in ("spread", "total", "moneyline"):
        print(f"\n{'='*65}")
        print(f"  MODEL TYPE: {model_type.upper()}")
        print(f"{'='*65}")

        X, y, base_weights = prepare_xy(df, model_type)

        if len(X) == 0:
            print(f"  No data for {model_type}, skipping.")
            continue

        # Apply tournament boosting to weights
        w_arr = base_weights.values.copy().astype(float)
        is_t = df.loc[X.index, "game_type"] == "tournament" if "game_type" in df.columns else np.zeros(len(X), bool)
        w_arr[is_t.values] *= 3.0
        boosted_weights = pd.Series(w_arr, index=X.index)

        print(f"  Training on {len(X)} samples "
              f"({int(is_t.sum())} tournament × 3× weight)")

        # Train models
        if model_type == "spread":
            models = train_spread_tournament(X, y, boosted_weights)
        elif model_type == "total":
            models = train_total_tournament(X, y, boosted_weights)
        else:
            models = train_moneyline_tournament(X, y, boosted_weights)

        # Print training results
        print(f"\n  Training results ({model_type}):")
        for name, data in models.items():
            if model_type == "moneyline":
                print(f"    {name:<25}: acc={data['accuracy']:.4f}  brier={data['brier']:.4f}")
            else:
                print(f"    {name:<25}: mae={data['mae']:.3f}  rmse={data['rmse']:.3f}")

        # Save
        save_tournament_models(models, model_type)

        # LOVO CV
        cv_results = lovo_cv_tournament(df, model_type)

        # Compare vs baseline on full tournament set
        print(f"\n  Comparison on all tournament games ({model_type}):")
        comparison = compare_with_baseline(df, models, model_type)
        for name, metrics in comparison.items():
            metrics_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
            print(f"    {name:<30}: {metrics_str}")

        all_results[model_type] = {
            "training": {n: {k: v for k, v in d.items() if k not in ("model", "scaler")}
                         for n, d in models.items()},
            "lovo_cv": cv_results.to_dict("records") if len(cv_results) > 0 else [],
            "comparison": comparison,
        }

    # Save consolidated report
    report_path = MODEL_DIR / "tournament_model_results.json"
    with open(report_path, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n{'='*65}")
    print(f"  Results saved -> {report_path}")
    print(f"  Tournament models saved -> {MODEL_DIR}/tournament_*.joblib")
    print(f"{'='*65}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tournament-specific prediction models")
    parser.add_argument("--feature-set", default="enriched",
                        choices=["enriched", "weighted", "comprehensive"],
                        help="Which training dataset to use")
    args = parser.parse_args()
    main(feature_set=args.feature_set)

