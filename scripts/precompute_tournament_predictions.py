#!/usr/bin/env python3
"""
Precompute 2026 NCAA Tournament game predictions.

Generates spread, moneyline, and total predictions for every first-round
(and First Four) matchup using:
  - Tournament models (tournament_*.joblib)
  - KenPom / BartTorvik efficiency data (local canonical CSVs)
  - Monte Carlo simulation results from bracket_2026.json

No CBBD API key is required.

Usage:
    python scripts/precompute_tournament_predictions.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_tools.efficiency_loader import EfficiencyDataLoader
from predictions import normalize_team_name

DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "precomputed_predictions"
BRACKETS_DIR = DATA_DIR / "precomputed_brackets"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Seed-based expected spreads (historical average margins)
SEED_SPREAD_LOOKUP = {
    (1, 16): -28.0, (2, 15): -20.0, (3, 14): -16.0, (4, 13): -12.5,
    (5, 12): -9.5,  (6, 11): -7.0,  (7, 10): -5.0,  (8, 9): -1.5,
    (1, 11): -18.0, (1, 16): -28.0,
}
# Historical total for tournament games by round (1st round avg ~143)
TOURNAMENT_EXPECTED_TOTAL = 143.5

# Historical upset rates
HISTORICAL_UPSET_RATES = {
    (1, 16): 0.01, (2, 15): 0.06, (3, 14): 0.15, (4, 13): 0.21,
    (5, 12): 0.35, (6, 11): 0.37, (7, 10): 0.39, (8, 9): 0.49,
}


def load_models() -> dict:
    """Load tournament prediction models."""
    models = {}
    TOURNEY_VARIANTS = {
        "moneyline": ["xgboost", "logistic_regression", "gradient_boosting"],
        "spread":    ["xgboost", "ridge", "random_forest"],
        "total":     ["xgboost", "ridge", "random_forest"],
    }
    for model_type, variants in TOURNEY_VARIANTS.items():
        models[model_type] = {}
        models[f"{model_type}_scalers"] = {}
        for variant in variants:
            mf = MODEL_DIR / f"tournament_{model_type}_{variant}.joblib"
            sf = MODEL_DIR / f"tournament_{model_type}_{variant}_scaler.joblib"
            if mf.exists():
                try:
                    models[model_type][variant] = joblib.load(mf)
                except Exception as exc:
                    print(f"  Warning: could not load {mf.name}: {exc}")
            if sf.exists():
                try:
                    models[f"{model_type}_scalers"][variant] = joblib.load(sf)
                except Exception:
                    pass
    loaded = {k: len(v) for k, v in models.items() if isinstance(v, dict)}
    print(f"  Loaded tournament models: {loaded}")
    return models


def load_efficiency_data():
    """Load KenPom and BartTorvik local data."""
    loader = EfficiencyDataLoader()
    kenpom_df = None
    bart_df = None
    try:
        kenpom_df = loader.load_kenpom()
        print(f"  KenPom: {len(kenpom_df)} teams")
    except Exception as exc:
        print(f"  Warning: KenPom load failed: {exc}")
    try:
        bart_df = loader.load_barttorvik()
        print(f"  BartTorvik: {len(bart_df)} teams")
    except Exception as exc:
        print(f"  Warning: BartTorvik load failed: {exc}")
    return kenpom_df, bart_df


def _build_espn_to_kenpom_map() -> dict:
    """Build a lookup dict: ESPN display name -> KenPom team row index."""
    map_file = DATA_DIR / "kenpom_to_espn_matches.csv"
    if not map_file.exists():
        return {}
    df = pd.read_csv(map_file)
    # kenpom column = KenPom name, espn_match = canonical ESPN form (may have " (fuzzy)")
    result = {}
    for _, row in df.iterrows():
        espn_raw = str(row.get("espn_match", "")).replace(" (fuzzy)", "").strip()
        kp_name = str(row.get("kenpom", "")).strip()
        if espn_raw and kp_name:
            result[espn_raw.lower()] = kp_name
    return result


ESPN_TO_KENPOM = _build_espn_to_kenpom_map()


def get_team_stats(name: str, kenpom_df: pd.DataFrame, bart_df: pd.DataFrame) -> dict:
    """Look up efficiency stats for a team by ESPN display name.

    Lookup order:
    1. normalize_team_name(name) against canonical_team (stripped of " (fuzzy)")
    2. exact ESPN display name against canonical_team
    3. ESPN -> KenPom name lookup via kenpom_to_espn_matches.csv
    4. Direct KenPom 'Team' column lookup
    """
    stats = {"name": name}
    norm_name = normalize_team_name(name)

    def _find_kenpom(df: pd.DataFrame) -> pd.Series | None:
        if df is None or df.empty:
            return None
        # Best: search via canonical_team (strip "(fuzzy)")
        if "canonical_team" in df.columns:
            # Strip (fuzzy) suffix for comparison
            canon_clean = df["canonical_team"].str.replace(r"\s*\(fuzzy\)$", "", regex=True).str.strip()
            for lookup in [norm_name, name]:
                mask = canon_clean == lookup
                if mask.any():
                    return df[mask].iloc[0]
        # Try the KenPom 'Team' column directly
        team_col = "Team" if "Team" in df.columns else None
        if team_col:
            for lookup in [norm_name, name]:
                mask = df[team_col] == lookup
                if mask.any():
                    return df[mask].iloc[0]
        # Use the ESPN->KenPom reverse lookup
        kenpom_name = ESPN_TO_KENPOM.get(norm_name.lower()) or ESPN_TO_KENPOM.get(name.lower())
        if kenpom_name and team_col:
            mask = df[team_col] == kenpom_name
            if mask.any():
                return df[mask].iloc[0]
        return None

    if kenpom_df is not None:
        row = _find_kenpom(kenpom_df)
        if row is not None:
            stats.update({
                "net_efficiency": float(row.get("NetRtg", 0) or 0),
                "off_efficiency": float(row.get("ORtg", 0) or 0),
                "def_efficiency": float(row.get("DRtg", 0) or 0),
                "tempo": float(row.get("AdjT", 70) or 70),
                "luck": float(row.get("Luck", 0) or 0),
                "sos": float(row.get("SOS_NetRtg", 0) or 0),
            })

    if bart_df is not None:
        row = _find_kenpom(bart_df)  # same logic works for barttorvik canonical_team
        if row is not None:
            # BartTorvik CSV has no headers; AdjOE=col1, AdjDE=col2 by convention.
            # Named lookups won't work — use positional access instead.
            try:
                adj_oe = float(row.iloc[1])
                adj_de = float(row.iloc[2])
            except (IndexError, ValueError, TypeError):
                adj_oe = adj_de = 0.0
            if adj_oe and adj_oe > 50:  # sanity check: realistic range 85-130
                stats.update({
                    "bart_adj_oe": adj_oe,
                    "bart_adj_de": adj_de,
                })

    return stats


def build_spread_features(home_stats: dict, away_stats: dict) -> dict:
    """Build spread/moneyline feature dict — matches the 18 trained features."""
    def s(d, k, default=0.0):
        v = d.get(k, default)
        return float(v) if v is not None else float(default)

    home_net = s(home_stats, "net_efficiency")
    away_net = s(away_stats, "net_efficiency")
    home_off = s(home_stats, "off_efficiency", 105)
    away_off = s(away_stats, "off_efficiency", 105)
    home_def = s(home_stats, "def_efficiency", 100)
    away_def = s(away_stats, "def_efficiency", 100)
    home_oe  = s(home_stats, "bart_adj_oe", home_off)
    away_oe  = s(away_stats, "bart_adj_oe", away_off)
    home_de  = s(home_stats, "bart_adj_de", home_def)
    away_de  = s(away_stats, "bart_adj_de", away_def)
    home_tempo = s(home_stats, "tempo", 70)
    away_tempo = s(away_stats, "tempo", 70)

    return {
        # CBBD-derived game stats — not available without API key; set to 0
        "spread_net_rating_diff":  home_net - away_net,
        "spread_off_rating_diff":  home_off - away_off,
        "spread_def_rating_diff":  home_def - away_def,
        "spread_ppg_diff":         0.0,
        "spread_opp_ppg_diff":     0.0,
        "spread_margin_diff":      0.0,
        "spread_efg_diff":         0.0,
        "spread_to_rate_diff":     0.0,
        "spread_orb_diff":         0.0,
        "spread_ft_rate_diff":     0.0,
        # KenPom features
        "kenpom_netrtg_diff": home_net - away_net,
        "kenpom_ortg_diff":   home_off - away_off,
        "kenpom_drtg_diff":   home_def - away_def,
        "kenpom_adjt_diff":   home_tempo - away_tempo,
        "kenpom_luck_diff":   s(home_stats, "luck") - s(away_stats, "luck"),
        "kenpom_sos_diff":    s(home_stats, "sos")  - s(away_stats, "sos"),
        # BartTorvik features
        "bart_oe_diff": home_oe - away_oe,
        "bart_de_diff": home_de - away_de,
    }


def build_total_features(home_stats: dict, away_stats: dict) -> dict:
    """Build total/over-under feature dict — matches the 19 trained features."""
    def s(d, k, default=0.0):
        v = d.get(k, default)
        return float(v) if v is not None else float(default)

    home_net = s(home_stats, "net_efficiency")
    away_net = s(away_stats, "net_efficiency")
    home_off = s(home_stats, "off_efficiency", 105)
    away_off = s(away_stats, "off_efficiency", 105)
    home_def = s(home_stats, "def_efficiency", 100)
    away_def = s(away_stats, "def_efficiency", 100)
    home_oe  = s(home_stats, "bart_adj_oe", home_off)
    away_oe  = s(away_stats, "bart_adj_oe", away_off)
    home_de  = s(home_stats, "bart_adj_de", home_def)
    away_de  = s(away_stats, "bart_adj_de", away_def)
    home_tempo = s(home_stats, "tempo", 70)
    away_tempo = s(away_stats, "tempo", 70)
    avg_tempo = (home_tempo + away_tempo) / 2 or 68.0

    # Rough projected total using pace × average scoring rate
    projected_total = round((home_oe + away_oe) * avg_tempo / 100, 1)

    # Estimate PPG: AdjOE (pts/100 poss) × AdjT (poss/game) / 100
    home_ppg     = home_off * home_tempo / 100
    away_ppg     = away_off * away_tempo / 100
    home_opp_ppg = home_def * away_tempo / 100
    away_opp_ppg = away_def * home_tempo / 100

    return {
        "total_combined_off_eff":  home_off + away_off,
        "total_combined_def_eff":  home_def + away_def,
        "total_avg_off_eff":       (home_off + away_off) / 2,
        "total_avg_def_eff":       (home_def + away_def) / 2,
        "total_combined_tempo":    home_tempo + away_tempo,
        "total_avg_tempo":         avg_tempo,
        "total_combined_ppg":      home_ppg + away_ppg,
        "total_combined_opp_ppg":  home_opp_ppg + away_opp_ppg,
        "total_combined_fg_pct":   0.90,   # 0.45 × 2 teams (tournament average)
        "total_combined_3pt_pct":  0.68,   # 0.34 × 2 teams (tournament average)
        "total_projected_total":   projected_total,
        # KenPom features
        "kenpom_netrtg_diff": home_net - away_net,
        "kenpom_ortg_diff":   home_off - away_off,
        "kenpom_drtg_diff":   home_def - away_def,
        "kenpom_adjt_diff":   home_tempo - away_tempo,
        "kenpom_luck_diff":   s(home_stats, "luck") - s(away_stats, "luck"),
        "kenpom_sos_diff":    s(home_stats, "sos")  - s(away_stats, "sos"),
        # BartTorvik features
        "bart_oe_diff": home_oe - away_oe,
        "bart_de_diff": home_de - away_de,
    }


def _predict_with_model(model, scaler, feat_dict: dict) -> float | None:
    """Run one model variant using a named feature dict."""
    try:
        import pandas as pd
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            cols = list(feat_dict.keys())
        x_df = pd.DataFrame([feat_dict])[cols]
        if scaler is not None:
            x_df = pd.DataFrame(
                scaler.transform(x_df), columns=cols, index=x_df.index
            )
        return float(model.predict(x_df)[0])
    except Exception:
        return None


def _predict_proba_with_model(model, scaler, feat_dict: dict) -> float | None:
    """Run one moneyline model; return probability of home team winning."""
    try:
        import pandas as pd
        if hasattr(model, "feature_names_in_"):
            cols = list(model.feature_names_in_)
        else:
            cols = list(feat_dict.keys())
        x_df = pd.DataFrame([feat_dict])[cols]
        if scaler is not None:
            x_df = pd.DataFrame(
                scaler.transform(x_df), columns=cols, index=x_df.index
            )
        return float(model.predict_proba(x_df)[0][1])
    except Exception:
        return None


def predict_game(
    home_name: str, home_seed: int,
    away_name: str, away_seed: int,
    home_stats: dict, away_stats: dict,
    models: dict,
) -> dict:
    """Run all models on a single matchup and return consensus prediction."""

    if not home_stats.get("net_efficiency") and not away_stats.get("net_efficiency"):
        # No efficiency data: fall back to seed model
        seed_diff = away_seed - home_seed
        win_prob = 1 / (1 + np.exp(-seed_diff * 0.4))
        spread_range = SEED_SPREAD_LOOKUP.get(
            (min(home_seed, away_seed), max(home_seed, away_seed)), 0
        )
        if home_seed > away_seed:
            spread_range = -spread_range
        return {
            "home_win_prob": float(win_prob),
            "away_win_prob": float(1 - win_prob),
            "predicted_spread": float(spread_range),
            "confidence_interval_spread": 8.0,
            "predicted_total": TOURNAMENT_EXPECTED_TOTAL,
            "confidence_interval_total": 10.0,
            "method": "seed_fallback",
        }

    spread_feats = build_spread_features(home_stats, away_stats)
    total_feats  = build_total_features(home_stats, away_stats)

    # ---- Spread ----
    spread_preds = []
    for variant, mdl in models.get("spread", {}).items():
        scaler = models.get("spread_scalers", {}).get(variant)
        val = _predict_with_model(mdl, scaler, spread_feats)
        if val is not None:
            spread_preds.append(val)

    pred_spread = float(np.median(spread_preds)) if spread_preds else (
        SEED_SPREAD_LOOKUP.get((min(home_seed, away_seed), max(home_seed, away_seed)), 0)
    )
    ci_spread = float(np.std(spread_preds) * 1.96) if len(spread_preds) > 1 else 8.0

    # ---- Total ----
    total_preds = []
    for variant, mdl in models.get("total", {}).items():
        scaler = models.get("total_scalers", {}).get(variant)
        val = _predict_with_model(mdl, scaler, total_feats)
        if val is not None:
            total_preds.append(val)

    pred_total = float(np.median(total_preds)) if total_preds else TOURNAMENT_EXPECTED_TOTAL
    ci_total = float(np.std(total_preds) * 1.96) if len(total_preds) > 1 else 10.0

    # ---- Moneyline ----
    win_prob_preds = []
    for variant, mdl in models.get("moneyline", {}).items():
        scaler = models.get("moneyline_scalers", {}).get(variant)
        prob = _predict_proba_with_model(mdl, scaler, spread_feats)
        if prob is not None:
            win_prob_preds.append(prob)

    home_win_prob = float(np.mean(win_prob_preds)) if win_prob_preds else (
        1 / (1 + np.exp(-0.15 * (home_stats.get("net_efficiency", 0) - away_stats.get("net_efficiency", 0))))

    )

    # ---- Upset probability vs historical rate ----
    fav_seed = min(home_seed, away_seed)
    dog_seed = max(home_seed, away_seed)
    hist_rate = HISTORICAL_UPSET_RATES.get((fav_seed, dog_seed), 0.5)
    # Model-based upset prob = probability that higher seed wins
    if home_seed > away_seed:
        model_upset_prob = home_win_prob
    else:
        model_upset_prob = 1 - home_win_prob

    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
        "predicted_spread": round(pred_spread, 1),
        "confidence_interval_spread": round(ci_spread, 1),
        "predicted_total": round(pred_total, 1),
        "confidence_interval_total": round(ci_total, 1),
        "historical_upset_rate": round(hist_rate, 3),
        "model_upset_prob": round(model_upset_prob, 4),
        "upset_signal": model_upset_prob > hist_rate + 0.05,
        "method": "tournament_model" if win_prob_preds else "efficiency_model",
    }


def main():
    print("=" * 70)
    print("PRECOMPUTING 2026 NCAA TOURNAMENT PREDICTIONS")
    print("=" * 70)

    # Load resources
    print("\nLoading models...")
    models = load_models()

    print("\nLoading efficiency data...")
    kenpom_df, bart_df = load_efficiency_data()

    # Load the bracket
    bracket_file = BRACKETS_DIR / "bracket_2026.json"
    if not bracket_file.exists():
        print(f"ERROR: {bracket_file} not found. Run scripts/fetch_2026_bracket.py first.")
        return
    with open(bracket_file) as f:
        bracket = json.load(f)

    teams_list = bracket["bracket_data"]["teams"]
    sim_results = bracket["simulation_results"]

    # Build team id -> sim result map
    sim_map = {
        t["name"]: {
            "seed": t["seed"],
            "region": t["region"],
            "winner_prob": sim_results.get(tid, {}).get("winner_prob", 0),
            "final_four_prob": sim_results.get(tid, {}).get("final_four_prob", 0),
            "championship_prob": sim_results.get(tid, {}).get("championship_prob", 0),
        }
        for tid, t in [(k, v["team"]) for k, v in sim_results.items()]
    }

    # Fetch ESPN Round of 64 + First Four games from the API
    print("\nFetching game schedule from ESPN...")
    import requests
    from datetime import timedelta

    all_matchups = []
    seen_events = set()

    for offset in range(7):
        dt = datetime(2026, 3, 17) + timedelta(days=offset)
        date_str = dt.strftime("%Y%m%d")
        try:
            r = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/"
                "mens-college-basketball/scoreboard",
                params={"dates": date_str, "limit": 100},
                timeout=15,
            )
        except Exception:
            continue
        if r.status_code != 200:
            continue
        events = r.json().get("events", [])
        for e in events:
            if e["id"] in seen_events:
                continue
            comp = e.get("competitions", [{}])[0]
            notes = comp.get("notes", [{}])
            headline = notes[0].get("headline", "") if notes else ""
            if "NCAA" not in headline:
                continue
            seen_events.add(e["id"])

            region = "TBD"
            for reg in ["East", "West", "Midwest", "South"]:
                if reg in headline:
                    region = reg
                    break

            round_label = "1st Round"
            if "First Four" in headline:
                round_label = "First Four"
            elif "2nd Round" in headline:
                round_label = "2nd Round"

            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue

            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])

            home_name = home.get("team", {}).get("displayName", "TBD")
            away_name = away.get("team", {}).get("displayName", "TBD")

            # Get seeds from sim_map
            home_seed = sim_map.get(home_name, {}).get("seed", 0)
            away_seed = sim_map.get(away_name, {}).get("seed", 0)

            all_matchups.append({
                "event_id": e["id"],
                "date": e.get("date", ""),
                "game_name": e.get("name", ""),
                "region": region,
                "round_label": round_label,
                "headline": headline,
                "home_team": home_name,
                "away_team": away_name,
                "home_seed": home_seed,
                "away_seed": away_seed,
            })

    print(f"  Found {len(all_matchups)} tournament games")

    # Predict each game
    print("\nGenerating predictions...")
    predictions = []

    for matchup in all_matchups:
        home = matchup["home_team"]
        away = matchup["away_team"]

        if home == "TBD" or away == "TBD":
            continue

        home_stats = get_team_stats(home, kenpom_df, bart_df)
        away_stats = get_team_stats(away, kenpom_df, bart_df)

        pred = predict_game(
            home_name=home, home_seed=matchup["home_seed"],
            away_name=away, away_seed=matchup["away_seed"],
            home_stats=home_stats, away_stats=away_stats,
            models=models,
        )

        # Add simulation context
        home_sim = sim_map.get(home, {})
        away_sim = sim_map.get(away, {})

        entry = {
            **matchup,
            **pred,
            "home_winner_prob_sim": home_sim.get("winner_prob", 0),
            "home_final_four_prob": home_sim.get("final_four_prob", 0),
            "away_winner_prob_sim": away_sim.get("winner_prob", 0),
            "away_final_four_prob": away_sim.get("final_four_prob", 0),
            "home_kenpom": home_stats.get("net_efficiency", "N/A"),
            "away_kenpom": away_stats.get("net_efficiency", "N/A"),
        }
        predictions.append(entry)

    print(f"  Generated predictions for {len(predictions)} games")

    # Print summary
    print("\n" + "=" * 70)
    print("2026 NCAA TOURNAMENT PREDICTIONS SUMMARY")
    print("=" * 70)

    for r_name in ["First Four", "1st Round"]:
        round_games = [p for p in predictions if p["round_label"] == r_name]
        if not round_games:
            continue
        print(f"\n--- {r_name} ---")
        for p in sorted(round_games, key=lambda x: (x["region"], x["home_seed"])):
            fav = p["home_team"] if p["home_win_prob"] >= 0.5 else p["away_team"]
            fav_prob = max(p["home_win_prob"], p["away_win_prob"])
            spread = abs(p["predicted_spread"])
            upset = " ** UPSET ALERT **" if p.get("upset_signal") else ""
            print(
                f"  {p['region']:10} ({p['home_seed']:2}) {p['home_team']:28} vs "
                f"({p['away_seed']:2}) {p['away_team']:28}  "
                f"Fav: {fav} ({fav_prob:.0%})  Spread: {spread:.1f}  "
                f"Total: {p['predicted_total']:.0f}{upset}"
            )

    # Upset watch list
    print("\n--- UPSET WATCH LIST ---")
    upset_candidates = [
        p for p in predictions
        if p["round_label"] == "1st Round" and p.get("upset_signal")
    ]
    if upset_candidates:
        for p in sorted(upset_candidates, key=lambda x: x["model_upset_prob"], reverse=True):
            dog_team = p["home_team"] if p["home_seed"] > p["away_seed"] else p["away_team"]
            dog_seed = max(p["home_seed"], p["away_seed"])
            fav_seed = min(p["home_seed"], p["away_seed"])
            print(
                f"  ({dog_seed}) {dog_team:30} over ({fav_seed}) seed  "
                f"Model upset prob: {p['model_upset_prob']:.0%}  "
                f"Historical rate: {p['historical_upset_rate']:.0%}"
            )
    else:
        print("  No strong upset signals detected.")

    # Save to JSON
    today = datetime.now().strftime("%Y-%m-%d")
    out_file = OUTPUT_DIR / f"tournament_predictions_{today}.json"
    output = {
        "year": 2026,
        "generated_at": datetime.now().isoformat(),
        "games": predictions,
        "num_games": len(predictions),
    }
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved {len(predictions)} predictions to {out_file}")

    # Also save as upcoming_game_predictions.json (used by some pages)
    upcoming_file = DATA_DIR / "upcoming_game_predictions.json"
    with open(upcoming_file, "w") as f:
        json.dump(
            [
                {
                    "game_id": p["event_id"],
                    "date": p["date"],
                    "home_team": p["home_team"],
                    "away_team": p["away_team"],
                    "home_win_prob": p["home_win_prob"],
                    "away_win_prob": p["away_win_prob"],
                    "predicted_spread": p["predicted_spread"],
                    "predicted_total": p["predicted_total"],
                    "upset_signal": p.get("upset_signal", False),
                    "region": p["region"],
                    "round_label": p["round_label"],
                    "home_seed": p["home_seed"],
                    "away_seed": p["away_seed"],
                }
                for p in predictions
            ],
            f,
            indent=2,
            default=str,
        )
    print(f"Updated {upcoming_file}")
    print("\nDone!")


if __name__ == "__main__":
    main()
