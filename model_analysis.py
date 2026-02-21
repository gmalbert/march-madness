#!/usr/bin/env python3
"""Feature importance analysis and hyperparameter tuning for betting models.

Run modes
---------
python model_analysis.py                  # all analysis
python model_analysis.py --importance     # feature importance only
python model_analysis.py --tune           # hyperparameter tuning only
python model_analysis.py --compare        # baseline vs tournament model comparison

Outputs
-------
data_files/models/feature_importance.json
data_files/models/best_hyperparams.json
data_files/models/model_ab_test_results.json
"""

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd
import joblib


class _NpEncoder(json.JSONEncoder):
    """JSON encoder that converts numpy scalars to native Python types."""
    def default(self, obj: Any):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (accuracy_score, brier_score_loss,
                              mean_absolute_error, mean_squared_error)
from sklearn.model_selection import (GridSearchCV, RandomizedSearchCV,
                                      StratifiedKFold, KFold)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

warnings.filterwarnings("ignore")

DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_training_data(enriched: bool = True) -> pd.DataFrame:
    path = DATA_DIR / ("training_data_enriched.csv" if enriched else "training_data_weighted.csv")
    if not path.exists():
        path = DATA_DIR / "training_data_weighted.csv"
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path.name}")
    return df


def get_feature_set(df: pd.DataFrame, model_type: str) -> List[str]:
    if model_type in ("spread", "moneyline"):
        cols = [c for c in df.columns if c.startswith("spread_")]
    else:
        cols = [c for c in df.columns if c.startswith("total_")]
    kp_bt = [c for c in df.columns if c.startswith("kenpom_") or c.startswith("bart_")]
    return cols + kp_bt


def prep(df: pd.DataFrame, model_type: str):
    feat_cols = get_feature_set(df, model_type)
    feat_cols = [c for c in feat_cols if c in df.columns]

    target = "actual_spread" if model_type in ("spread", "moneyline") else "actual_total"
    valid = df.dropna(subset=[target]).copy()
    X = valid[feat_cols].fillna(0)

    if model_type == "moneyline":
        y = (valid[target] < 0).astype(int)
    else:
        y = valid[target]

    weights = valid.get("sample_weight",
                        pd.Series(np.ones(len(valid)), index=valid.index)).values
    return X, y, weights, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
# 1.  FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────

def analyse_feature_importance(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute XGBoost gain-based and permutation importance for each model type."""
    results: Dict[str, Any] = {}

    for model_type in ("spread", "total", "moneyline"):
        print(f"\n── Feature importance: {model_type} ──")
        X, y, w, feat_cols = prep(df, model_type)

        if len(X) < 50:
            print(f"  Skipping {model_type}: not enough data.")
            continue

        is_clf = model_type == "moneyline"

        # Fit a fresh XGB model on the full dataset for importance analysis
        if is_clf:
            model = XGBClassifier(n_estimators=200, max_depth=4,
                                   learning_rate=0.05, eval_metric="logloss",
                                   random_state=42)
        else:
            model = XGBRegressor(n_estimators=200, max_depth=4,
                                  learning_rate=0.05, random_state=42)

        model.fit(X, y, sample_weight=w)

        # --- Gain importance (built-in XGBoost) ---
        gain_imp = model.get_booster().get_score(importance_type="gain")
        # Map f-index back to column names when names aren't used
        if gain_imp and list(gain_imp.keys())[0].startswith("f"):
            gain_imp = {feat_cols[int(k[1:])]: v for k, v in gain_imp.items()
                        if int(k[1:]) < len(feat_cols)}
        total_gain = max(sum(gain_imp.values()), 1e-9)
        gain_pct = {k: round(v / total_gain * 100, 3) for k, v in
                    sorted(gain_imp.items(), key=lambda x: x[1], reverse=True)}

        print(f"  Top-10 gain importance (%):")
        for i, (feat, pct) in enumerate(list(gain_pct.items())[:10], 1):
            print(f"    {i:2}. {feat:<40} {pct:>6.2f}%")

        # --- Permutation importance (more reliable, model-agnostic) ---
        # Use a small held-out set (last 20% in time)
        split = int(0.8 * len(X))
        X_te, y_te = X.iloc[split:], y.iloc[split:]
        perm = permutation_importance(model, X_te, y_te,
                                       n_repeats=10, random_state=42,
                                       scoring="accuracy" if is_clf else "neg_mean_absolute_error")
        perm_means = perm.importances_mean
        perm_stds  = perm.importances_std
        perm_df = pd.DataFrame({
            "feature": feat_cols,
            "importance_mean": perm_means,
            "importance_std":  perm_stds,
        }).sort_values("importance_mean", ascending=False).reset_index(drop=True)
        perm_dict = perm_df.head(20).to_dict(orient="records")

        print(f"  Top-10 permutation importance:")
        for row in perm_df.head(10).itertuples():
            print(f"    {row.Index+1:2}. {row.feature:<40} {row.importance_mean:>8.5f} ± {row.importance_std:.5f}")

        results[model_type] = {
            "gain_importance_pct": gain_pct,
            "permutation_importance": perm_dict,
        }

    # Save
    out = MODEL_DIR / "feature_importance.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    print(f"\n✅ Feature importance saved → {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 2.  HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

PARAM_GRIDS = {
    "xgb_clf": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
    "xgb_reg": {
        "n_estimators":     [100, 200, 300],
        "max_depth":        [3, 4, 5],
        "learning_rate":    [0.03, 0.05, 0.1],
        "subsample":        [0.7, 0.8, 1.0],
        "colsample_bytree": [0.7, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
    },
}


def tune_hyperparameters(df: pd.DataFrame,
                          n_iter: int = 25,
                          cv: int = 5) -> Dict[str, Any]:
    """RandomizedSearch hyperparameter tuning for XGBoost models.

    Parameters
    ----------
    n_iter : number of random parameter settings to try per model type
    cv     : number of cross-validation folds
    """
    best_params: Dict[str, Any] = {}

    for model_type in ("spread", "total", "moneyline"):
        print(f"\n── Hyperparameter tuning: {model_type} ──")
        X, y, w, _ = prep(df, model_type)
        if len(X) < 100:
            print(f"  Not enough data, skipping.")
            continue

        is_clf = model_type == "moneyline"
        scoring = "accuracy" if is_clf else "neg_mean_absolute_error"
        cv_obj = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42) \
                 if is_clf else KFold(n_splits=cv, shuffle=True, random_state=42)

        if is_clf:
            base = XGBClassifier(eval_metric="logloss", random_state=42)
            grid = PARAM_GRIDS["xgb_clf"]
        else:
            base = XGBRegressor(random_state=42)
            grid = PARAM_GRIDS["xgb_reg"]

        search = RandomizedSearchCV(
            base, grid,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv_obj,
            random_state=42,
            n_jobs=-1,
        )
        search.fit(X, y, sample_weight=w)

        bp = search.best_params_
        bs = search.best_score_
        print(f"  Best score ({scoring}): {bs:.4f}")
        print(f"  Best params: {bp}")

        best_params[model_type] = {
            "best_score": round(float(bs), 6),
            "scoring": scoring,
            "best_params": bp,
        }

        # Retrain with best params and save
        if is_clf:
            best_model = XGBClassifier(**bp, eval_metric="logloss", random_state=42)
        else:
            best_model = XGBRegressor(**bp, random_state=42)
        best_model.fit(X, y, sample_weight=w)

        out_path = MODEL_DIR / f"tuned_{model_type}_xgboost.joblib"
        joblib.dump(best_model, out_path)
        print(f"  Saved tuned model → {out_path}")

    out = MODEL_DIR / "best_hyperparams.json"
    with open(out, "w") as fh:
        json.dump(best_params, fh, indent=2)
    print(f"\n✅ Best hyperparams saved → {out}")
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 3.  A/B TEST: regular-season vs tournament models on tournament hold-out
# ─────────────────────────────────────────────────────────────────────────────

def ab_test_models(df: pd.DataFrame) -> Dict[str, Any]:
    """Compare regular-season and tournament-specific models on tournament data.

    Tournament data is further split into a hold-out (last 2 years) vs
    historical to approximate out-of-sample evaluation.
    """
    print("\n── A/B Test: Regular vs Tournament Models ──")
    df_tourney = df[df["game_type"] == "tournament"].copy()
    if len(df_tourney) == 0:
        print("  No tournament data available.")
        return {}

    results: Dict[str, Any] = {}
    hold_years = sorted(df_tourney["season"].unique())[-2:]  # last 2 tournament years
    print(f"  Hold-out years: {hold_years}")

    for model_type in ("spread", "total", "moneyline"):
        print(f"\n  Model type: {model_type}")
        feat_cols = [c for c in df.columns
                     if c.startswith("spread_") or c.startswith("total_")
                     or c.startswith("kenpom_") or c.startswith("bart_")]
        feat_cols = [c for c in feat_cols if c in df.columns]
        if not feat_cols:
            print("  No features found, skipping.")
            continue

        target = "actual_spread" if model_type in ("spread", "moneyline") else "actual_total"
        df_valid = df_tourney.dropna(subset=[target])
        hold_mask = df_valid["season"].isin(hold_years)
        train_mask = ~hold_mask

        X_tr = df_valid.loc[train_mask, feat_cols].fillna(0)
        X_te = df_valid.loc[hold_mask, feat_cols].fillna(0)
        if model_type == "moneyline":
            y_tr = (df_valid.loc[train_mask, target] < 0).astype(int)
            y_te = (df_valid.loc[hold_mask, target] < 0).astype(int)
        else:
            y_tr = df_valid.loc[train_mask, target]
            y_te = df_valid.loc[hold_mask, target]

        if len(X_te) == 0:
            print("  No hold-out data.")
            continue

        is_clf = model_type == "moneyline"

        # ── Model A: all data, no tournament weighting ──────────────────
        if is_clf:
            model_A = XGBClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4, eval_metric="logloss", random_state=42)
        else:
            model_A = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, random_state=42)
        model_A.fit(X_tr, y_tr)

        # ── Model B: tournament data weighted 3× ───────────────────────
        w = np.ones(len(X_tr))
        is_t = df_valid.loc[train_mask, "game_type"] == "tournament"
        w[is_t.values] *= 3.0

        if is_clf:
            model_B = XGBClassifier(n_estimators=200, learning_rate=0.05,
                                     max_depth=4, eval_metric="logloss", random_state=42)
        else:
            model_B = XGBRegressor(n_estimators=200, learning_rate=0.05,
                                    max_depth=4, random_state=42)
        model_B.fit(X_tr, y_tr, sample_weight=w)

        # ── Also load the saved tournament model if available ───────────
        saved_path = MODEL_DIR / f"tournament_{model_type}_xgboost.joblib"
        model_C = None
        if saved_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    model_C = joblib.load(saved_path)
            except Exception as e:
                print(f"  Could not load saved tournament model: {e}")

        def _eval(m, X, y):
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    f = list(m.feature_names_in_) if hasattr(m, 'feature_names_in_') else None
                    if f:
                        Xm = pd.DataFrame(
                            np.zeros((len(X), len(f))), columns=f)
                        for c in X.columns:
                            if c in Xm.columns:
                                Xm[c] = X[c].values
                        preds = m.predict(Xm.values)
                    else:
                        preds = m.predict(X.values)
                except Exception:
                    preds = m.predict(X.values)
            if is_clf:
                return {"accuracy": round(accuracy_score(y, preds), 4),
                        "n_games": len(y)}
            else:
                return {"mae": round(mean_absolute_error(y, preds), 3),
                        "rmse": round(float(np.sqrt(mean_squared_error(y, preds))), 3),
                        "n_games": len(y)}

        res: Dict[str, Any] = {
            "hold_out_years": hold_years,
            "model_A_no_weighting": _eval(model_A, X_te, y_te),
            "model_B_tournament_weighted": _eval(model_B, X_te, y_te),
        }
        if model_C is not None:
            res["model_C_saved_tournament"] = _eval(model_C, X_te, y_te)

        # Print comparison table
        print(f"  {'Model':<35} {'Metric':>15}")
        for name, metrics in res.items():
            if isinstance(metrics, dict) and any(k in metrics for k in ('accuracy', 'mae')):
                m_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
                print(f"    {name:<35} {m_str}")

        results[model_type] = res

    # Save
    out = MODEL_DIR / "model_ab_test_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2, cls=_NpEncoder)
    print(f"\n✅ A/B test results saved → {out}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(run_importance: bool = True,
         run_tuning: bool = True,
         run_compare: bool = True,
         tune_iter: int = 20):
    print("\n" + "=" * 65)
    print("   MODEL ANALYSIS  (model_analysis.py)")
    print("=" * 65)

    df = load_training_data(enriched=True)

    if run_importance:
        importance_results = analyse_feature_importance(df)

    if run_tuning:
        print("\nRunning hyperparameter tuning "
              f"({tune_iter} random iterations per model type) …")
        tune_results = tune_hyperparameters(df, n_iter=tune_iter)

    if run_compare:
        ab_results = ab_test_models(df)

    print("\n" + "=" * 65)
    print("  Analysis complete.  Reports saved to data_files/models/")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model analysis utility")
    parser.add_argument("--importance", action="store_true",
                        help="Run feature importance only")
    parser.add_argument("--tune", action="store_true",
                        help="Run hyperparameter tuning only")
    parser.add_argument("--compare", action="store_true",
                        help="Run A/B comparison only")
    parser.add_argument("--iter", type=int, default=20,
                        help="Number of RandomizedSearchCV iterations (default 20)")
    args = parser.parse_args()

    if not any([args.importance, args.tune, args.compare]):
        main(tune_iter=args.iter)
    else:
        df = load_training_data(enriched=True)
        if args.importance:
            analyse_feature_importance(df)
        if args.tune:
            tune_hyperparameters(df, n_iter=args.iter)
        if args.compare:
            ab_test_models(df)
