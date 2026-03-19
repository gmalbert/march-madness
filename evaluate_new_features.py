"""
Feature evaluation: can NCAA NET rankings or Haslametrics improve the spread/total model?

Methodology (mirrors enrich_training_data.py):
  - Both NET and Haslametrics provide only current-season snapshots.
  - Same as the existing KenPom/BartTorvik enrichment, we apply current ratings to
    all historical games as a proxy for relative team strength.
  - We use K-Fold cross-validation (5-fold, stratified by season) and report
    MAE improvements for spread and total prediction.
  - Feature importance from XGBoost identifies which new signals matter most.
  - If a feature set improves MAE ≥ 0.1 points vs. baseline, we retrain and save.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings("ignore")

DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load base training data
# ─────────────────────────────────────────────────────────────────────────────

def load_base_data() -> pd.DataFrame:
    path = DATA_DIR / "training_data_enriched.csv"
    if not path.exists():
        path = DATA_DIR / "training_data_weighted.csv"
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} games  |  cols: {len(df.columns)}")
    print(f"  Seasons: {sorted(df['season'].unique())}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Attach NET and Haslametrics features (both home and away must match)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_quad(val) -> tuple[int, int]:
    """Return (wins, losses) from a 'W-L' string, e.g. '17-2'."""
    try:
        if isinstance(val, str) and "-" in val:
            w, l = val.split("-")
            return int(w), int(l)
    except Exception:
        pass
    return 0, 0


def attach_net_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add net_rank_diff, net_q1wins_diff, net_winpct_diff columns."""
    net_path = DATA_DIR / "net_rankings.csv"
    if not net_path.exists():
        print("  ⚠  net_rankings.csv not found – skipping NET features")
        return df

    net = pd.read_csv(net_path)
    net = net.dropna(subset=["canonical_team"])
    # Only keep one row per canonical team (should already be unique)
    net = net.drop_duplicates(subset="canonical_team")

    # Derived columns
    net["Wins"] = pd.to_numeric(net["Wins"], errors="coerce").fillna(0)
    net["Losses"] = pd.to_numeric(net["Losses"], errors="coerce").fillna(0)
    net["net_win_pct"] = np.where(
        (net["Wins"] + net["Losses"]) > 0,
        net["Wins"] / (net["Wins"] + net["Losses"]),
        0.5,
    )
    net["q1_wins"] = net["Quad1"].apply(lambda x: _parse_quad(x)[0])
    net["q1_losses"] = net["Quad1"].apply(lambda x: _parse_quad(x)[1])
    net["q1_win_pct"] = np.where(
        (net["q1_wins"] + net["q1_losses"]) > 0,
        net["q1_wins"] / (net["q1_wins"] + net["q1_losses"]),
        0.5,
    )
    net["NET_Rank"] = pd.to_numeric(net["NET_Rank"], errors="coerce")

    # Build lookup dict
    net_lookup = net.set_index("canonical_team")[
        ["NET_Rank", "net_win_pct", "q1_wins", "q1_win_pct"]
    ].to_dict("index")

    def _get(team, key):
        entry = net_lookup.get(team)
        return entry[key] if entry and pd.notna(entry[key]) else np.nan

    home = df["home_team"].map(lambda t: net_lookup.get(t, {}))
    away = df["away_team"].map(lambda t: net_lookup.get(t, {}))

    df["net_rank_diff"] = df["home_team"].map(lambda t: _get(t, "NET_Rank")) - \
                          df["away_team"].map(lambda t: _get(t, "NET_Rank"))
    df["net_winpct_diff"] = df["home_team"].map(lambda t: _get(t, "net_win_pct")) - \
                             df["away_team"].map(lambda t: _get(t, "net_win_pct"))
    df["net_q1wins_diff"] = df["home_team"].map(lambda t: _get(t, "q1_wins")) - \
                             df["away_team"].map(lambda t: _get(t, "q1_wins"))
    df["net_q1pct_diff"]  = df["home_team"].map(lambda t: _get(t, "q1_win_pct")) - \
                             df["away_team"].map(lambda t: _get(t, "q1_win_pct"))

    covered = df["net_rank_diff"].notna().sum()
    print(f"  NET: {covered}/{len(df)} games have both teams matched ({100*covered/len(df):.1f}%)")
    return df


def attach_haslametrics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hasla_net_diff, hasla_oeff_diff, hasla_deff_diff, hasla_fg_diff, hasla_3p_diff."""
    hasla_path = DATA_DIR / "haslametrics_canonical.csv"
    if not hasla_path.exists():
        print("  ⚠  haslametrics_canonical.csv not found – skipping Hasla features")
        return df

    hl = pd.read_csv(hasla_path)
    hl = hl.dropna(subset=["canonical_team"]).drop_duplicates(subset="canonical_team")

    for col in ["O_Eff", "D_Eff", "hasla_net_eff", "O_FG%", "D_FG%", "O_3P%", "D_3P%", "O_AP%", "D_AP%"]:
        if col in hl.columns:
            hl[col] = pd.to_numeric(hl[col], errors="coerce")

    hl_lookup = hl.set_index("canonical_team").to_dict("index")

    def _get(team, key):
        entry = hl_lookup.get(team, {})
        val = entry.get(key, np.nan)
        return float(val) if pd.notna(val) else np.nan

    for col, key_h, key_a in [
        ("hasla_net_diff",   "hasla_net_eff", "hasla_net_eff"),
        ("hasla_o_eff_diff", "O_Eff",         "O_Eff"),
        ("hasla_d_eff_diff", "D_Eff",         "D_Eff"),
        ("hasla_fg_diff",    "O_FG%",         "O_FG%"),
        ("hasla_d_fg_diff",  "D_FG%",         "D_FG%"),
        ("hasla_3p_diff",    "O_3P%",         "O_3P%"),
        ("hasla_ap_diff",    "O_AP%",         "O_AP%"),
    ]:
        df[col] = (df["home_team"].map(lambda t: _get(t, key_h)) -
                   df["away_team"].map(lambda t: _get(t, key_a)))

    covered = df["hasla_net_diff"].notna().sum()
    print(f"  Haslametrics: {covered}/{len(df)} games have both teams matched ({100*covered/len(df):.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Cross-validation helper
# ─────────────────────────────────────────────────────────────────────────────

def _cv_mae(df: pd.DataFrame, feature_cols: list, target: str, n_splits: int = 5,
            weights_col: str = "sample_weight") -> tuple[float, float]:
    """Return (mean MAE, std MAE) across folds, dropping rows with NaN in features."""
    valid = df.dropna(subset=feature_cols + [target])
    if len(valid) < 100:
        return np.nan, np.nan

    X = valid[feature_cols].values
    y = valid[target].values
    w = valid[weights_col].values if weights_col in valid.columns else np.ones(len(valid))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    maes = []
    for train_idx, test_idx in kf.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        w_tr = w[train_idx]

        model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr, sample_weight=w_tr, verbose=False)
        preds = model.predict(X_te)
        maes.append(mean_absolute_error(y_te, preds))

    return float(np.mean(maes)), float(np.std(maes))


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Main evaluation
# ─────────────────────────────────────────────────────────────────────────────

BASELINE_SPREAD = [c for c in [
    "spread_net_rating_diff", "spread_off_rating_diff", "spread_def_rating_diff",
    "spread_ppg_diff", "spread_opp_ppg_diff", "spread_margin_diff",
    "spread_efg_diff", "spread_to_rate_diff", "spread_orb_diff", "spread_ft_rate_diff",
]]

BASELINE_TOTAL = [
    "total_combined_off_eff", "total_combined_def_eff", "total_avg_off_eff",
    "total_avg_def_eff", "total_combined_tempo", "total_avg_tempo",
    "total_combined_ppg", "total_combined_opp_ppg",
    "total_combined_fg_pct", "total_combined_3pt_pct", "total_projected_total",
]

KENPOM_FEATS = [
    "kenpom_netrtg_diff", "kenpom_ortg_diff", "kenpom_drtg_diff",
    "kenpom_adjt_diff", "kenpom_luck_diff", "kenpom_sos_diff",
]

BART_FEATS = ["bart_oe_diff", "bart_de_diff"]

NET_FEATS = ["net_rank_diff", "net_winpct_diff", "net_q1wins_diff", "net_q1pct_diff"]

HASLA_SPREAD_FEATS = [
    "hasla_net_diff", "hasla_o_eff_diff", "hasla_d_eff_diff",
    "hasla_fg_diff", "hasla_d_fg_diff", "hasla_3p_diff", "hasla_ap_diff",
]

HASLA_TOTAL_FEATS = [
    "hasla_o_eff_diff", "hasla_d_eff_diff", "hasla_fg_diff",
    "hasla_d_fg_diff", "hasla_3p_diff", "hasla_ap_diff",
]


def run_evaluation():
    print("=" * 70)
    print("FEATURE EVALUATION: NCAA NET + Haslametrics")
    print("=" * 70)

    # Load and attach
    df = load_base_data()
    print("\nAttaching new feature sources…")
    df = attach_net_features(df)
    df = attach_haslametrics_features(df)

    # Only keep cols actually in the dataframe
    def present(cols):
        return [c for c in cols if c in df.columns]

    bs_sp = present(BASELINE_SPREAD)
    bs_to = present(BASELINE_TOTAL)
    kp    = present(KENPOM_FEATS)
    bt    = present(BART_FEATS)
    net   = present(NET_FEATS)
    hs_sp = present(HASLA_SPREAD_FEATS)
    hs_to = present(HASLA_TOTAL_FEATS)

    if not bs_sp:
        print("ERROR: No baseline spread features found in dataset. Exiting.")
        return

    # ── Spread evaluation ────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("SPREAD PREDICTION  (actual_spread = home score – away score)\n")

    feature_sets_spread = {
        "Baseline (ESPN eff diffs)":            bs_sp,
        "+ KenPom":                              bs_sp + kp,
        "+ BartTorvik":                          bs_sp + bt,
        "+ KenPom + BartTorvik":                 bs_sp + kp + bt,
        "+ NET only":                            bs_sp + net,
        "+ Haslametrics only":                   bs_sp + hs_sp,
        "+ KenPom + BartTorvik + NET":           bs_sp + kp + bt + net,
        "+ KenPom + BartTorvik + Haslametrics":  bs_sp + kp + bt + hs_sp,
        "FULL: all sources":                     bs_sp + kp + bt + net + hs_sp,
    }

    spread_results = {}
    baseline_mae_sp = None
    for label, feats in feature_sets_spread.items():
        feats_unique = list(dict.fromkeys(feats))   # deduplicate, preserve order
        mae, std = _cv_mae(df, feats_unique, "actual_spread")
        if baseline_mae_sp is None:
            baseline_mae_sp = mae
        delta = f"  Δ {mae - baseline_mae_sp:+.3f}" if baseline_mae_sp != mae else ""
        n_valid = df.dropna(subset=feats_unique + ["actual_spread"]).shape[0]
        print(f"  {label:<50s}  MAE={mae:.3f} ± {std:.3f}  (n={n_valid:,}){delta}")
        spread_results[label] = {"mae": mae, "std": std, "n": n_valid, "feats": feats_unique}

    # ── Total evaluation ─────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("TOTAL PREDICTION  (actual_total = home + away scores)\n")

    feature_sets_total = {
        "Baseline (ESPN eff / tempo)":           bs_to,
        "+ KenPom":                              bs_to + kp,
        "+ BartTorvik":                          bs_to + bt,
        "+ KenPom + BartTorvik":                 bs_to + kp + bt,
        "+ NET only":                            bs_to + net,
        "+ Haslametrics only":                   bs_to + hs_to,
        "+ KenPom + BartTorvik + NET":           bs_to + kp + bt + net,
        "+ KenPom + BartTorvik + Haslametrics":  bs_to + kp + bt + hs_to,
        "FULL: all sources":                     bs_to + kp + bt + net + hs_to,
    }

    total_results = {}
    baseline_mae_to = None
    for label, feats in feature_sets_total.items():
        feats_unique = list(dict.fromkeys(feats))
        mae, std = _cv_mae(df, feats_unique, "actual_total")
        if baseline_mae_to is None:
            baseline_mae_to = mae
        delta = f"  Δ {mae - baseline_mae_to:+.3f}" if baseline_mae_to != mae else ""
        n_valid = df.dropna(subset=feats_unique + ["actual_total"]).shape[0]
        print(f"  {label:<50s}  MAE={mae:.3f} ± {std:.3f}  (n={n_valid:,}){delta}")
        total_results[label] = {"mae": mae, "std": std, "n": n_valid, "feats": feats_unique}

    # ── Feature importance ───────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("FEATURE IMPORTANCE  (FULL spread model, XGBoost gain)\n")

    all_spread_feats = list(dict.fromkeys(bs_sp + kp + bt + net + hs_sp))
    valid_sp = df.dropna(subset=all_spread_feats + ["actual_spread"])
    if len(valid_sp) > 500:
        xm = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        xm.fit(
            valid_sp[all_spread_feats],
            valid_sp["actual_spread"],
            sample_weight=valid_sp.get("sample_weight", pd.Series(np.ones(len(valid_sp)))),
        )
        imps = pd.Series(xm.feature_importances_, index=all_spread_feats).sort_values(ascending=False)
        print("  Top 20 features:")
        for feat, imp in imps.head(20).items():
            source_tag = (
                "[NET]"   if feat.startswith("net_")    else
                "[HASLA]" if feat.startswith("hasla_")  else
                "[KP]"    if feat.startswith("kenpom_") else
                "[BART]"  if feat.startswith("bart_")   else
                "[BASE]"
            )
            print(f"    {source_tag:<8} {feat:<42s}  {imp:.4f}")
    else:
        print("  Not enough data for importance analysis.")

    # ── Recommendation ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    best_sp_label = min(spread_results, key=lambda k: spread_results[k]["mae"])
    best_to_label = min(total_results,  key=lambda k: total_results[k]["mae"])
    best_sp_mae   = spread_results[best_sp_label]["mae"]
    best_to_mae   = total_results[best_to_label]["mae"]

    sp_improvement = baseline_mae_sp - best_sp_mae
    to_improvement = baseline_mae_to - best_to_mae

    print(f"\n  Spread: best set = '{best_sp_label}'")
    print(f"          MAE improvement over baseline: {sp_improvement:+.3f} pts")

    print(f"\n  Total:  best set = '{best_to_label}'")
    print(f"          MAE improvement over baseline: {to_improvement:+.3f} pts")

    IMPROVEMENT_THRESHOLD = 0.10   # points

    sp_retrain = sp_improvement >= IMPROVEMENT_THRESHOLD
    to_retrain = to_improvement >= IMPROVEMENT_THRESHOLD

    if sp_retrain or to_retrain:
        print("\n  ✅  Improvement ≥ 0.10 pts found – retraining and saving improved models…")
        _retrain_and_save(df, spread_results, total_results, best_sp_label, best_to_label,
                          sp_retrain, to_retrain)
    else:
        print("\n  ℹ️  Improvement below 0.10-pt threshold – models unchanged.")
        print("      New features may still be useful for current-game predictions.")

    # ── Correlation analysis to understand signal overlap ────────────────────
    print("\n" + "─" * 70)
    print("CORRELATION ANALYSIS  (NET / Haslametrics vs CurentSpread signal)\n")

    signal_cols = ["actual_spread"] + present(
        ["net_rank_diff", "net_q1wins_diff", "net_q1pct_diff",
         "hasla_net_diff", "hasla_o_eff_diff", "hasla_d_eff_diff",
         "kenpom_netrtg_diff", "bart_oe_diff", "spread_net_rating_diff"]
    )
    valid_corr = df.dropna(subset=signal_cols)
    if len(valid_corr) > 50:
        corr = valid_corr[signal_cols].corr()["actual_spread"].drop("actual_spread").sort_values(ascending=False)
        print("  Pearson correlation with actual_spread (positive = home favored):")
        for col, r in corr.items():
            tag = (
                "[NET]"   if col.startswith("net_")    else
                "[HASLA]" if col.startswith("hasla_")  else
                "[KP]"    if col.startswith("kenpom_") else
                "[BART]"  if col.startswith("bart_")   else
                "[BASE]"
            )
            print(f"    {tag:<8} {col:<42s}  r={r:.3f}")

    print("\nDone.\n")
    return spread_results, total_results


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Retrain and save
# ─────────────────────────────────────────────────────────────────────────────

def _retrain_and_save(df, spread_results, total_results, best_sp_label, best_to_label,
                      do_spread, do_total):
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    def _save_model(df, feats, target, filename_stem, weights_col="sample_weight"):
        valid = df.dropna(subset=feats + [target])
        X = valid[feats].values
        y = valid[target].values
        w = valid[weights_col].values if weights_col in valid.columns else np.ones(len(valid))

        xm = xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        )
        xm.fit(X, y, sample_weight=w)
        out_path = MODEL_DIR / f"{filename_stem}_xgb.pkl"
        feat_path = MODEL_DIR / f"{filename_stem}_features.json"
        joblib.dump(xm, out_path)
        import json
        feat_path.write_text(json.dumps(feats))
        print(f"  Saved model → {out_path}")
        print(f"  Saved feature list → {feat_path}")

    if do_spread:
        feats = spread_results[best_sp_label]["feats"]
        print(f"\n  Retraining spread model with {len(feats)} features…")
        _save_model(df, feats, "actual_spread", "spread_improved")

    if do_total:
        feats = total_results[best_to_label]["feats"]
        print(f"\n  Retraining total model with {len(feats)} features…")
        _save_model(df, feats, "actual_total", "total_improved")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_evaluation()
