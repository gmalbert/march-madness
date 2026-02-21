#!/usr/bin/env python3
"""
Generate predictions for upcoming college basketball games.
This script loads upcoming games, fetches team data, and generates predictions.

Tournament detection: if a game's season_type contains 'postseason' / 'tournament'
(case-insensitive) the tournament-specific models (tournament_*_xgboost.joblib)
are used instead of the regular-season advanced models.  Those models were trained
with tournament games weighted 3× and use the full enriched feature set.

Confidence intervals are estimated as ±1σ of the ensemble spread across all
model variants.  For classification, confidence is the max class probability.
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import fetch_team_stats, fetch_efficiency_ratings, fetch_adjusted_efficiency, fetch_betting_lines
from features import (
    calculate_efficiency_differential,
    calculate_spread_features,
    calculate_total_features,
    calculate_win_probability_features,
    project_game_total,
)
from fetch_live_odds import fetch_live_odds

# Import normalize_team_name from predictions.py for consistency
from predictions import normalize_team_name

# Configuration
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"

# Remove the local normalize_team_name function since we're importing it

def load_models():
    """Load trained prediction models (regular-season + tournament-specific)."""
    models = {}

    # Load advanced models first (these are the primary regular-season models)
    advanced_models = ['moneyline_advanced', 'spread_advanced', 'total_advanced']
    for model_name in advanced_models:
        model_file = MODEL_DIR / f"{model_name}.joblib"
        if model_file.exists():
            try:
                model_key = model_name.replace('_advanced', '')
                models[f'{model_key}_advanced'] = joblib.load(model_file)
                print(f"Loaded advanced {model_key} model")
            except Exception as e:
                print(f"Error loading advanced {model_name}: {e}")

    # ------------------------------------------------------------------ #
    # Tournament-specific models (trained with 3× tournament weighting)   #
    # ------------------------------------------------------------------ #
    TOURNEY_VARIANTS = {
        'moneyline': ['xgboost', 'logistic_regression', 'gradient_boosting'],
        'spread':    ['xgboost', 'ridge', 'random_forest'],
        'total':     ['xgboost', 'ridge', 'random_forest'],
    }
    for model_type, variants in TOURNEY_VARIANTS.items():
        models[f'tournament_{model_type}'] = {}
        models[f'tournament_{model_type}_scalers'] = {}
        for variant in variants:
            mf = MODEL_DIR / f"tournament_{model_type}_{variant}.joblib"
            if mf.exists():
                try:
                    models[f'tournament_{model_type}'][variant] = joblib.load(mf)
                    print(f"Loaded tournament {model_type} {variant}")
                except Exception as e:
                    print(f"Error loading tournament {model_type} {variant}: {e}")
            sf = MODEL_DIR / f"tournament_{model_type}_{variant}_scaler.joblib"
            if sf.exists():
                try:
                    models[f'tournament_{model_type}_scalers'][variant] = joblib.load(sf)
                except Exception as e:
                    pass

    # Load basic models as fallback (suppress warnings for old versions)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

        model_types = ['spread', 'total', 'moneyline']
        for model_type in model_types:
            models[model_type] = {}
            models[f'{model_type}_scalers'] = {}

            # Try to load each model variant
            for variant in ['xgboost', 'random_forest', 'linear_regression', 'logistic_regression']:
                model_file = MODEL_DIR / f"{model_type}_{variant}.joblib"
                if model_file.exists():
                    try:
                        models[model_type][variant] = joblib.load(model_file)
                        print(f"Loaded {model_type} {variant} model")
                    except Exception as e:
                        print(f"Error loading {model_type} {variant}: {e}")

            # Load scalers for linear/logistic models
            scaler_file = MODEL_DIR / f"{model_type}_linear_regression_scaler.joblib"
            if scaler_file.exists():
                try:
                    models[f'{model_type}_scalers']['linear'] = joblib.load(scaler_file)
                    print(f"Loaded {model_type} linear scaler")
                except Exception as e:
                    print(f"Error loading {model_type} scaler: {e}")

    return models

def calculate_features(home_stats, away_stats, home_eff, away_eff):
    """Calculate prediction features from team data using `features.py` helpers.

    Falls back to the original minimal efficiency-diff features if richer
    fields aren't available.
    """
    # Try to compute efficiency differentials using features helper
    try:
        eff_diff = calculate_efficiency_differential(home_eff or {}, away_eff or {})
    except Exception:
        # Fallback to legacy keys
        home_off = (home_eff.get('adj_off') if home_eff else None) or (home_eff.get('offensiveRating') if home_eff else None) or 100
        away_off = (away_eff.get('adj_off') if away_eff else None) or (away_eff.get('offensiveRating') if away_eff else None) or 100
        home_def = (home_eff.get('adj_def') if home_eff else None) or (home_eff.get('defensiveRating') if home_eff else None) or 100
        away_def = (away_eff.get('adj_def') if away_eff else None) or (away_eff.get('defensiveRating') if away_eff else None) or 100
        eff_diff = {
            'off_eff_diff': float(home_off) - float(away_off),
            'def_eff_diff': float(home_def) - float(away_def),
            'net_eff_diff': (float(home_off) - float(home_def)) - (float(away_off) - float(away_def))
        }

    # Build spread/total/moneyline feature dicts using helpers when possible
    try:
        spread_feats = calculate_spread_features(home_stats or {}, away_stats or {}, home_eff or {}, away_eff or {})
    except Exception:
        spread_feats = {
            'net_rating_diff': eff_diff.get('net_eff_diff', 0),
            'off_rating_diff': eff_diff.get('off_eff_diff', 0),
            'def_rating_diff': eff_diff.get('def_eff_diff', 0),
            'ppg_diff': (home_stats or {}).get('ppg', 0) - (away_stats or {}).get('ppg', 0),
            'opp_ppg_diff': (home_stats or {}).get('opp_ppg', 0) - (away_stats or {}).get('opp_ppg', 0),
            'margin_diff': 0,
            'efg_diff': (home_stats or {}).get('efg_pct', 0) - (away_stats or {}).get('efg_pct', 0),
            'to_rate_diff': (home_stats or {}).get('to_rate', 0) - (away_stats or {}).get('to_rate', 0),
            'orb_diff': (home_stats or {}).get('orb_pct', 0) - (away_stats or {}).get('orb_pct', 0),
            'ft_rate_diff': (home_stats or {}).get('ft_rate', 0) - (away_stats or {}).get('ft_rate', 0),
        }

    try:
        total_feats = calculate_total_features(home_stats or {}, away_stats or {}, home_eff or {}, away_eff or {})
    except Exception:
        total_feats = {
            'combined_tempo': (home_stats or {}).get('pace', 70) + (away_stats or {}).get('pace', 70),
            'avg_tempo': ((home_stats or {}).get('pace', 70) + (away_stats or {}).get('pace', 70)) / 2,
            'combined_ppg': (home_stats or {}).get('ppg', 0) + (away_stats or {}).get('ppg', 0),
            'combined_opp_ppg': (home_stats or {}).get('opp_ppg', 0) + (away_stats or {}).get('opp_ppg', 0),
            'combined_off_eff': (home_eff or {}).get('offensiveRating', 0) + (away_eff or {}).get('offensiveRating', 0),
            'combined_def_eff': (home_eff or {}).get('defensiveRating', 0) + (away_eff or {}).get('defensiveRating', 0),
            'projected_total': project_game_total(home_eff or {}, away_eff or {})
        }

    # Moneyline features: reuse spread_feats + win-prob helpers when available
    try:
        win_feats = calculate_win_probability_features(home_stats or {}, away_stats or {})
    except Exception:
        win_feats = {'net_rating_diff': eff_diff.get('net_eff_diff', 0)}

    # Also include the original 3-feature vector for backward compatibility
    minimal = {
        'off_eff_diff': eff_diff.get('off_eff_diff', 0),
        'def_eff_diff': eff_diff.get('def_eff_diff', 0),
        'net_eff_diff': eff_diff.get('net_eff_diff', 0)
    }

    return {
        'spread': {**minimal, **spread_feats},
        'total': {**minimal, **total_feats},
        'moneyline': {**minimal, **win_feats}
    }

def _build_tournament_feature_vector(features: dict, model) -> np.ndarray:
    """Build a feature vector aligned to the tournament model's expected columns.

    Tournament models were trained on spread_* / total_* columns plus kenpom_*/bart_*
    enriched features.  When enriched data is unavailable at inference, those columns
    are filled with 0 (neutral – no edge from that feature).
    """
    try:
        expected_cols = list(model.feature_names_in_)
    except AttributeError:
        try:
            n = model.n_features_in_
            expected_cols = [f"f{i}" for i in range(n)]
        except AttributeError:
            return None

    row = {}
    # Populate from available features (spread / total / moneyline dicts)
    all_feat = {}
    for v in features.values():
        if isinstance(v, dict):
            all_feat.update(v)

    for col in expected_cols:
        # Strip 'spread_' or 'total_' prefix to look up in feature dict
        stripped = col.replace("spread_", "").replace("total_", "")
        row[col] = float(all_feat.get(col, all_feat.get(stripped, 0)) or 0)

    return np.array([[row[c] for c in expected_cols]])


def make_predictions(game_data: dict, models: dict,
                     is_tournament: bool = False) -> dict:
    """Make predictions for a game using trained models.

    Parameters
    ----------
    game_data : dict
        Game info including home/away team efficiency and stats.
    models : dict
        Loaded model dictionary from load_models().
    is_tournament : bool
        If True, use tournament-specific models and widen confidence intervals
        to reflect higher variance in tournament games.

    Returns
    -------
    dict
        Prediction results including spread, total, win probabilities, and
        confidence-interval estimates.
    """
    features = calculate_features(
        game_data.get('home_stats'), game_data.get('away_stats'),
        game_data.get('home_eff'), game_data.get('away_eff')
    )

    predictions = {}

    # ── baseline 3-feature vector (all models understand this) ──────────
    feature_names = ['off_eff_diff', 'def_eff_diff', 'net_eff_diff']
    base_feature_df = pd.DataFrame([features['spread']], columns=feature_names)

    # ── helper: run multiple variants and collect predictions ────────────
    def _run_variants(model_dict: dict, scaler_dict: dict,
                      feature_df: pd.DataFrame, kind: str) -> list:
        preds = []
        for name, m in model_dict.items():
            try:
                sc = scaler_dict.get(name)
                X = sc.transform(feature_df) if sc else feature_df.values
                if kind == "clf":
                    p = m.predict_proba(X)[0][1]
                else:
                    p = float(m.predict(X)[0])
                preds.append(p)
            except Exception:
                pass
        return preds

    # ── TOURNAMENT MODELS ────────────────────────────────────────────────
    if is_tournament:
        t_spread = models.get('tournament_spread', {})
        t_total  = models.get('tournament_total', {})
        t_ml     = models.get('tournament_moneyline', {})
        t_spread_sc = models.get('tournament_spread_scalers', {})
        t_total_sc  = models.get('tournament_total_scalers', {})
        t_ml_sc     = models.get('tournament_moneyline_scalers', {})

        # Build enriched feature vectors for tournament models
        if t_spread:
            best = t_spread.get('xgboost') or next(iter(t_spread.values()), None)
            if best:
                vec = _build_tournament_feature_vector(features, best)
                if vec is not None:
                    preds_s: list = []
                    for name, m in t_spread.items():
                        try:
                            sc = t_spread_sc.get(name)
                            X = sc.transform(
                                pd.DataFrame(vec, columns=list(best.feature_names_in_))) \
                                if sc else vec
                            preds_s.append(float(m.predict(X)[0]))
                        except Exception:
                            pass
                    if preds_s:
                        predictions['spread_prediction'] = float(np.mean(preds_s))
                        predictions['spread_confidence_interval'] = [
                            round(float(np.mean(preds_s) - np.std(preds_s)), 2),
                            round(float(np.mean(preds_s) + np.std(preds_s)), 2),
                        ]
                        predictions['model_source'] = 'tournament'

        if t_total:
            best = t_total.get('xgboost') or next(iter(t_total.values()), None)
            if best:
                vec = _build_tournament_feature_vector(features, best)
                if vec is not None:
                    preds_t: list = []
                    for name, m in t_total.items():
                        try:
                            sc = t_total_sc.get(name)
                            X = sc.transform(
                                pd.DataFrame(vec, columns=list(best.feature_names_in_))) \
                                if sc else vec
                            preds_t.append(float(m.predict(X)[0]))
                        except Exception:
                            pass
                    if preds_t:
                        predictions['total_prediction'] = float(np.mean(preds_t))
                        predictions['total_confidence_interval'] = [
                            round(float(np.mean(preds_t) - np.std(preds_t)), 2),
                            round(float(np.mean(preds_t) + np.std(preds_t)), 2),
                        ]

        if t_ml:
            best = t_ml.get('xgboost') or next(iter(t_ml.values()), None)
            if best:
                vec = _build_tournament_feature_vector(features, best)
                if vec is not None:
                    preds_m: list = []
                    for name, m in t_ml.items():
                        try:
                            sc = t_ml_sc.get(name)
                            X = sc.transform(
                                pd.DataFrame(vec, columns=list(best.feature_names_in_))) \
                                if sc else vec
                            preds_m.append(float(m.predict_proba(X)[0][1]))
                        except Exception:
                            pass
                    if preds_m:
                        avg = float(np.mean(preds_m))
                        std = float(np.std(preds_m)) if len(preds_m) > 1 else 0.05
                        predictions['moneyline_home_win_prob'] = avg
                        predictions['moneyline_away_win_prob'] = 1.0 - avg
                        predictions['moneyline_confidence_interval'] = [
                            round(max(0.0, avg - std), 4),
                            round(min(1.0, avg + std), 4),
                        ]

        # If all tournament model predictions succeeded, return early
        if ('spread_prediction' in predictions and
                'total_prediction' in predictions and
                'moneyline_home_win_prob' in predictions):
            return predictions

    # ── ADVANCED MODELS (regular-season, 3-feature) ──────────────────────
    adv_preds_spread: list = []
    adv_preds_total:  list = []
    adv_preds_ml:     list = []

    if models.get('spread_advanced'):
        try:
            p = float(models['spread_advanced'].predict(base_feature_df)[0])
            predictions.setdefault('spread_prediction', p)
            adv_preds_spread.append(p)
        except Exception as e:
            print(f"Error with advanced spread model: {e}")

    if models.get('total_advanced'):
        try:
            p = float(models['total_advanced'].predict(base_feature_df)[0])
            predictions.setdefault('total_prediction', p)
            adv_preds_total.append(p)
        except Exception as e:
            print(f"Error with advanced total model: {e}")

    if models.get('moneyline_advanced'):
        try:
            proba = models['moneyline_advanced'].predict_proba(base_feature_df)[0]
            predictions.setdefault('moneyline_home_win_prob', float(proba[1]))
            predictions.setdefault('moneyline_away_win_prob', float(proba[0]))
            adv_preds_ml.append(float(proba[1]))
        except Exception as e:
            print(f"Error with advanced moneyline model: {e}")

    # ── FALLBACK: ensemble of basic models ────────────────────────────────
    if not models.get('spread_advanced') and models.get('spread'):
        preds = _run_variants(models['spread'], models.get('spread_scalers', {}),
                              base_feature_df, 'reg')
        if preds:
            predictions['spread_prediction'] = float(np.mean(preds))
            adv_preds_spread.extend(preds)

    if not models.get('total_advanced') and models.get('total'):
        preds = _run_variants(models['total'], models.get('total_scalers', {}),
                              base_feature_df, 'reg')
        if preds:
            predictions['total_prediction'] = float(np.mean(preds))
            adv_preds_total.extend(preds)

    if not models.get('moneyline_advanced') and models.get('moneyline'):
        preds = _run_variants(models['moneyline'], models.get('moneyline_scalers', {}),
                              base_feature_df, 'clf')
        if preds:
            avg = float(np.mean(preds))
            predictions['moneyline_home_win_prob'] = avg
            predictions['moneyline_away_win_prob'] = 1.0 - avg
            adv_preds_ml.extend(preds)

    # ── CONFIDENCE INTERVALS ─────────────────────────────────────────────
    # Also include the basic-model variants in ensemble spread for CI
    if models.get('spread'):
        preds = _run_variants(models['spread'], models.get('spread_scalers', {}),
                              base_feature_df, 'reg')
        adv_preds_spread.extend(preds)
    if models.get('total'):
        preds = _run_variants(models['total'], models.get('total_scalers', {}),
                              base_feature_df, 'reg')
        adv_preds_total.extend(preds)
    if models.get('moneyline'):
        preds = _run_variants(models['moneyline'], models.get('moneyline_scalers', {}),
                              base_feature_df, 'clf')
        adv_preds_ml.extend(preds)

    if 'spread_prediction' in predictions and not predictions.get('spread_confidence_interval'):
        if len(adv_preds_spread) > 1:
            mu, sigma = np.mean(adv_preds_spread), np.std(adv_preds_spread)
        else:
            mu = predictions['spread_prediction']
            sigma = abs(mu) * 0.15 if mu != 0 else 3.0
        predictions['spread_confidence_interval'] = [
            round(float(mu - sigma), 2), round(float(mu + sigma), 2)]

    if 'total_prediction' in predictions and not predictions.get('total_confidence_interval'):
        if len(adv_preds_total) > 1:
            mu, sigma = np.mean(adv_preds_total), np.std(adv_preds_total)
        else:
            mu = predictions['total_prediction']
            sigma = abs(mu) * 0.05 if mu != 0 else 5.0
        predictions['total_confidence_interval'] = [
            round(float(mu - sigma), 2), round(float(mu + sigma), 2)]

    if 'moneyline_home_win_prob' in predictions and not predictions.get('moneyline_confidence_interval'):
        if len(adv_preds_ml) > 1:
            mu, sigma = np.mean(adv_preds_ml), np.std(adv_preds_ml)
        else:
            mu = predictions['moneyline_home_win_prob']
            sigma = 0.05
        predictions['moneyline_confidence_interval'] = [
            round(max(0.0, float(mu - sigma)), 4),
            round(min(1.0, float(mu + sigma)), 4)]

    predictions.setdefault('model_source', 'regular')
    return predictions

def load_team_data_once():
    """Load team efficiency and stats data once for all games."""
    current_year = 2025  # Use 2025 season data for current predictions

    print("Loading team data for all games...")
    efficiency_data = fetch_efficiency_ratings(current_year)
    team_stats_data = fetch_team_stats(current_year)

    # Create lookup dictionaries
    efficiency_lookup = {team['team']: team for team in efficiency_data}
    stats_lookup = {team['team']: team for team in team_stats_data}

    print(f"Loaded data for {len(efficiency_data)} teams")
    return efficiency_lookup, stats_lookup

def load_upcoming_games():
    """Load upcoming games from the current season CSV."""
    games_file = DATA_DIR / "espn_cbb_current_season.csv"
    if not games_file.exists():
        print(f"No upcoming games file found: {games_file}")
        return []

    df = pd.read_csv(games_file)

    # Filter to upcoming games (not completed)
    upcoming = df[df['status'] == 'STATUS_SCHEDULED'].copy()

    print(f"Found {len(upcoming)} upcoming games")

    return upcoming.to_dict('records')

def fetch_game_data(game, efficiency_lookup, stats_lookup, lines_data=None, live_odds=None):
    """Fetch team stats and efficiency data for a game using pre-loaded data."""
    current_year = 2025  # Use 2025 season data for current predictions

    try:
        # Get data for both teams
        home_team = game['home_team']
        away_team = game['away_team']

        # Clean team names for matching (remove common suffixes)
        def clean_team_name(name):
            return (name.replace(' Bonnies', '').replace(' Billikens', '')
                       .replace(' Wolverines', '').replace(' Buckeyes', ''))

        home_clean = clean_team_name(home_team)
        away_clean = clean_team_name(away_team)

        # Get efficiency data with fallbacks
        home_eff = efficiency_lookup.get(home_clean)
        away_eff = efficiency_lookup.get(away_clean)

        # If no efficiency data, create reasonable defaults based on rankings
        if not home_eff:
            home_rank = game.get('home_rank', 50)
            home_eff = {
                'offensiveRating': 110 - (home_rank / 10),  # Better teams have higher offensive rating
                'defensiveRating': 110 - (home_rank / 10)   # Better teams have better defense
            }
        else:
            # Keep original efficiency data structure
            home_eff = {
                'offensiveRating': home_eff.get('offensiveRating', 100),
                'defensiveRating': home_eff.get('defensiveRating', 100)
            }

        if not away_eff:
            away_rank = game.get('away_rank', 50)
            away_eff = {
                'offensiveRating': 110 - (away_rank / 10),
                'defensiveRating': 110 - (away_rank / 10)
            }
        else:
            # Keep original efficiency data structure
            away_eff = {
                'offensiveRating': away_eff.get('offensiveRating', 100),
                'defensiveRating': away_eff.get('defensiveRating', 100)
            }

        game_data = {
            'game_id': game.get('event_id'),
            'home_team': home_team,
            'away_team': away_team,
            'date': game.get('date'),
            'venue': game.get('venue'),
            'home_rank': game.get('home_rank'),
            'away_rank': game.get('away_rank'),
            'home_eff': home_eff,
            'away_eff': away_eff,
            'home_stats': stats_lookup.get(home_clean),
            'away_stats': stats_lookup.get(away_clean)
        }

        # If betting lines were provided, try to attach moneylines for market comparisons
        if lines_data:
            try:
                home_ml = None
                away_ml = None
                for line in lines_data:
                    # support dict or object
                    if isinstance(line, dict):
                        line_home = line.get('homeTeam') or line.get('home_team') or line.get('home')
                        line_away = line.get('awayTeam') or line.get('away_team') or line.get('away')
                        providers = line.get('lines') or []
                    else:
                        line_home = (getattr(line, 'home_team', None) or getattr(line, 'homeTeam', None) or getattr(line, 'home', None))
                        line_away = (getattr(line, 'away_team', None) or getattr(line, 'awayTeam', None) or getattr(line, 'away', None))
                        providers = getattr(line, 'lines', []) or []

                    try:
                        lh = str(line_home) if line_home is not None else None
                        la = str(line_away) if line_away is not None else None
                    except Exception:
                        lh = line_home
                        la = line_away

                    # Match by literal or cleaned names
                    if ((lh == home_team and la == away_team) or
                        (lh == home_clean and la == away_clean) or
                        (lh == normalize_team_name(home_team) and la == normalize_team_name(away_team))):
                        provider = providers[0] if providers and len(providers) > 0 else None
                        if provider:
                            if isinstance(provider, dict):
                                home_ml = provider.get('homeMoneyline') or provider.get('home_moneyline') or provider.get('homeML')
                                away_ml = provider.get('awayMoneyline') or provider.get('away_moneyline') or provider.get('awayML')
                            else:
                                home_ml = getattr(provider, 'homeMoneyline', None) or getattr(provider, 'home_moneyline', None) or getattr(provider, 'homeML', None)
                                away_ml = getattr(provider, 'awayMoneyline', None) or getattr(provider, 'away_moneyline', None) or getattr(provider, 'awayML', None)
                        else:
                            if isinstance(line, dict):
                                home_ml = line.get('homeMoneyline') or line.get('home_moneyline') or home_ml
                                away_ml = line.get('awayMoneyline') or line.get('away_moneyline') or away_ml
                            else:
                                home_ml = getattr(line, 'homeMoneyline', None) or getattr(line, 'home_moneyline', None) or home_ml
                                away_ml = getattr(line, 'awayMoneyline', None) or getattr(line, 'away_moneyline', None) or away_ml
                        break

                game_data['home_moneyline'] = home_ml
                game_data['away_moneyline'] = away_ml
                game_data['home_ml'] = home_ml
                game_data['away_ml'] = away_ml
            except Exception:
                # tolerate missing lines
                game_data['home_moneyline'] = None
                game_data['away_moneyline'] = None
                game_data['home_ml'] = None
                game_data['away_ml'] = None

        # If no moneylines from CFBD, try live odds
        if not game_data.get('home_moneyline') and live_odds:
            game_key = f"{normalize_team_name(home_team)} vs {normalize_team_name(away_team)}"
            # Also try the reverse order in case Odds API has home/away swapped
            reverse_key = f"{normalize_team_name(away_team)} vs {normalize_team_name(home_team)}"

            odds = None
            if game_key in live_odds:
                odds = live_odds[game_key]
            elif reverse_key in live_odds:
                # If found with reversed order, we need to swap the odds too
                odds = live_odds[reverse_key]
                # Swap home/away odds since the teams are swapped
                if odds:
                    odds = odds.copy()
                    # Swap moneyline
                    home_ml = odds.get('home_moneyline')
                    away_ml = odds.get('away_moneyline')
                    odds['home_moneyline'] = away_ml
                    odds['away_moneyline'] = home_ml
                    # Swap spread (flip the sign)
                    home_spread = odds.get('home_spread')
                    away_spread = odds.get('away_spread')
                    if home_spread is not None and away_spread is not None:
                        odds['home_spread'] = -away_spread  # Flip sign
                        odds['away_spread'] = -home_spread  # Flip sign
                        # Keep the odds the same since spread direction changed
                        home_spread_odds = odds.get('home_spread_odds')
                        away_spread_odds = odds.get('away_spread_odds')
                        odds['home_spread_odds'] = away_spread_odds
                        odds['away_spread_odds'] = home_spread_odds

            if odds:
                game_data['home_moneyline'] = odds.get('home_moneyline')
                game_data['away_moneyline'] = odds.get('away_moneyline')
                game_data['home_ml'] = odds.get('home_moneyline')
                game_data['away_ml'] = odds.get('away_moneyline')
                # Optionally add spread/total if needed
                game_data['home_spread'] = odds.get('home_spread')
                game_data['away_spread'] = odds.get('away_spread')
                game_data['total_line'] = odds.get('total_line')

        return game_data

    except Exception as e:
        print(f"Error fetching data for {game.get('home_team')} vs {game.get('away_team')}: {e}")
        return None

def refresh_espn_data():
    """Refresh ESPN game data to ensure we have the latest schedule."""
    print("Refreshing ESPN game data...")
    import subprocess
    import sys
    
    try:
        # Run the ESPN fetch script
        result = subprocess.run(
            [sys.executable, 'fetch_espn_cbb_scores.py'],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("ESPN data refreshed successfully")
            # Print summary from the fetch script
            if 'Total games' in result.stdout:
                for line in result.stdout.split('\n'):
                    if 'Total games' in line or 'Summary' in line:
                        print(f"  {line.strip()}")
            return True
        else:
            print(f"Warning: ESPN refresh had issues (exit code {result.returncode})")
            print("Continuing with existing data...")
            return False
    except Exception as e:
        print(f"Warning: Could not refresh ESPN data: {e}")
        print("Continuing with existing data...")
        return False

def generate_predictions():
    """Main function to generate predictions for all upcoming games."""
    print("Generating predictions for upcoming games...")
    print()
    
    # Refresh ESPN data first to get latest schedule
    refresh_espn_data()
    print()

    # Load models
    models = load_models()
    if not models:
        print("No models found. Please run model training first.")
        return

    # Load team data once for all games
    efficiency_lookup, stats_lookup = load_team_data_once()

    # Fetch betting lines once for the period (used to attach moneylines)
    try:
        lines_data = fetch_betting_lines(2025, 'postseason')
        print(f"Loaded {len(lines_data)} betting line entries")
    except Exception as e:
        print(f"Could not load betting lines: {e}")
        lines_data = None

    # Fetch live odds for current games
    try:
        live_odds = fetch_live_odds()
        print(f"Loaded live odds for {len(live_odds)} games")
    except Exception as e:
        print(f"Could not load live odds: {e}")
        live_odds = None

    # Load upcoming games
    upcoming_games = load_upcoming_games()
    if not upcoming_games:
        print("No upcoming games found.")
        return

    all_predictions = []
    successful_predictions = 0

    for game in upcoming_games:
        print(f"Predicting: {game['away_team']} @ {game['home_team']}")

        # Fetch game data using pre-loaded team data (and attach moneylines)
        game_data = fetch_game_data(game, efficiency_lookup, stats_lookup, lines_data=lines_data, live_odds=live_odds)
        if not game_data:
            print(f"  Could not fetch data for this game")
            continue

        # Detect tournament game
        season_type = str(game.get('season_type', '')).lower()
        is_tournament = any(kw in season_type for kw in ('postseason', 'tournament', 'ncaa'))

        # Make predictions (tournament-aware)
        predictions = make_predictions(game_data, models, is_tournament=is_tournament)
        if not predictions:
            print(f"  Could not generate predictions for this game")
            continue

        model_source = predictions.get('model_source', 'regular')
        ci_spread = predictions.get('spread_confidence_interval', [])
        print(f"  Generated predictions  [model={model_source}]"
              + (f"  spread CI={ci_spread}" if ci_spread else ""))

        # Combine game data with predictions
        game_result = {
            'game_info': game_data,
            'predictions': predictions,
            'is_tournament': is_tournament,
            'generated_at': datetime.now().isoformat(),
            'season': 2025
        }

        all_predictions.append(game_result)
        successful_predictions += 1

    # Save predictions
    output_file = DATA_DIR / "upcoming_game_predictions.json"
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2, default=str)

    print(f"\nSaved predictions for {successful_predictions}/{len(upcoming_games)} games to {output_file}")

    # Summary
    print("\nPrediction Summary:")
    print(f"  Total games: {len(upcoming_games)}")
    print(f"  Successful predictions: {successful_predictions}")
    print(f"  Models used: {len([k for k in models.keys() if not k.endswith('_scalers')])}")

if __name__ == "__main__":
    generate_predictions()