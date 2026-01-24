# March Madness Betting Predictions UI

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
from typing import Dict, List, Optional

# Configuration
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"

# Load models
@st.cache_resource
def load_models():
    """Load trained prediction models."""
    models = {}

    # Spread models
    try:
        models['spread'] = {
            'xgboost': joblib.load(MODEL_DIR / 'spread_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'spread_random_forest.joblib'),
            'linear': joblib.load(MODEL_DIR / 'spread_linear_regression.joblib')
        }
    except:
        st.error("Spread models not found. Please run model training first.")
        models['spread'] = None

    # Total models
    try:
        models['total'] = {
            'xgboost': joblib.load(MODEL_DIR / 'total_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'total_random_forest.joblib'),
            'linear': joblib.load(MODEL_DIR / 'total_linear_regression.joblib')
        }
    except:
        st.error("Total models not found. Please run model training first.")
        models['total'] = None

    # Moneyline models
    try:
        models['moneyline'] = {
            'xgboost': joblib.load(MODEL_DIR / 'moneyline_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'moneyline_random_forest.joblib'),
            'logistic': joblib.load(MODEL_DIR / 'moneyline_logistic_regression.joblib')
        }
    except:
        st.error("Moneyline models not found. Please run model training first.")
        models['moneyline'] = None

    return models

def get_upcoming_games() -> List[Dict]:
    """Get upcoming games (placeholder - would integrate with API)."""
    # For demo purposes, return some sample games
    return [
        {
            "home_team": "Duke",
            "away_team": "North Carolina",
            "home_eff": {"offensiveRating": 118.5, "defensiveRating": 89.2, "netRating": 29.3},
            "away_eff": {"offensiveRating": 115.8, "defensiveRating": 91.5, "netRating": 24.3},
            "home_stats": {"ppg": 78.5, "pace": 68.2, "efg_pct": 0.52, "to_rate": 0.15, "orb_pct": 0.32, "ft_rate": 0.35, "opp_ppg": 65.2},
            "away_stats": {"ppg": 75.8, "pace": 67.8, "efg_pct": 0.51, "to_rate": 0.16, "orb_pct": 0.31, "ft_rate": 0.33, "opp_ppg": 68.1},
            "betting_spread": -3.5,
            "betting_over_under": 145.5
        },
        {
            "home_team": "Kansas",
            "away_team": "Texas",
            "home_eff": {"offensiveRating": 122.1, "defensiveRating": 87.8, "netRating": 34.3},
            "away_eff": {"offensiveRating": 119.4, "defensiveRating": 90.1, "netRating": 29.3},
            "home_stats": {"ppg": 82.1, "pace": 69.5, "efg_pct": 0.54, "to_rate": 0.14, "orb_pct": 0.34, "ft_rate": 0.37, "opp_ppg": 63.8},
            "away_stats": {"ppg": 79.4, "pace": 68.9, "efg_pct": 0.53, "to_rate": 0.15, "orb_pct": 0.33, "ft_rate": 0.36, "opp_ppg": 66.2},
            "betting_spread": -4.0,
            "betting_over_under": 148.0
        }
    ]

def calculate_features(home_team: Dict, away_team: Dict, home_eff: Dict, away_eff: Dict) -> Dict:
    """Calculate prediction features for a game."""
    # Efficiency features
    off_eff_diff = home_eff.get("offensiveRating", 0) - away_eff.get("offensiveRating", 0)
    def_eff_diff = home_eff.get("defensiveRating", 0) - away_eff.get("defensiveRating", 0)
    net_eff_diff = home_eff.get("netRating", 0) - away_eff.get("netRating", 0)

    # Stats features
    ppg_diff = home_team.get("ppg", 0) - away_team.get("ppg", 0)
    opp_ppg_diff = home_team.get("opp_ppg", 0) - away_team.get("opp_ppg", 0)
    margin_diff = (home_team.get("ppg", 0) - home_team.get("opp_ppg", 0)) - (away_team.get("ppg", 0) - away_team.get("opp_ppg", 0))

    efg_diff = home_team.get("efg_pct", 0) - away_team.get("efg_pct", 0)
    to_rate_diff = home_team.get("to_rate", 0) - away_team.get("to_rate", 0)
    orb_diff = home_team.get("orb_pct", 0) - away_team.get("orb_pct", 0)
    ft_rate_diff = home_team.get("ft_rate", 0) - away_team.get("ft_rate", 0)

    # Total features
    combined_off_eff = home_eff.get("offensiveRating", 0) + away_eff.get("offensiveRating", 0)
    combined_def_eff = home_eff.get("defensiveRating", 0) + away_eff.get("defensiveRating", 0)
    avg_off_eff = (home_eff.get("offensiveRating", 0) + away_eff.get("offensiveRating", 0)) / 2
    avg_def_eff = (home_eff.get("defensiveRating", 0) + away_eff.get("defensiveRating", 0)) / 2

    combined_tempo = home_team.get("pace", 0) + away_team.get("pace", 0)
    avg_tempo = (home_team.get("pace", 0) + away_team.get("pace", 0)) / 2
    combined_ppg = home_team.get("ppg", 0) + away_team.get("ppg", 0)
    combined_opp_ppg = home_team.get("opp_ppg", 0) + away_team.get("opp_ppg", 0)
    combined_fg_pct = home_team.get("fg_pct", 0) + away_team.get("fg_pct", 0)
    combined_3pt_pct = home_team.get("three_pct", 0) + away_team.get("three_pct", 0)

    # Projected total
    projected_total = (avg_off_eff + avg_def_eff) / 2 * (avg_tempo / 100) * 0.8

    return {
        # Spread features
        'spread': [
            off_eff_diff, def_eff_diff, net_eff_diff,
            net_eff_diff, off_eff_diff, def_eff_diff,  # Duplicated for the model format
            ppg_diff, opp_ppg_diff, margin_diff,
            efg_diff, to_rate_diff, orb_diff, ft_rate_diff
        ],

        # Total features
        'total': [
            combined_off_eff, combined_def_eff, avg_off_eff, avg_def_eff,
            combined_tempo, avg_tempo, combined_ppg, combined_opp_ppg,
            combined_fg_pct, combined_3pt_pct, projected_total
        ],

        # Moneyline features (same as spread)
        'moneyline': [
            off_eff_diff, def_eff_diff, net_eff_diff,
            net_eff_diff, off_eff_diff, def_eff_diff,
            ppg_diff, opp_ppg_diff, margin_diff,
            efg_diff, to_rate_diff, orb_diff, ft_rate_diff
        ]
    }

def make_predictions(game_data: Dict, models: Dict) -> Dict:
    """Make predictions for a game using trained models."""
    features = calculate_features(
        game_data['home_stats'], game_data['away_stats'],
        game_data['home_eff'], game_data['away_eff']
    )

    predictions = {}

    # Spread predictions
    if models.get('spread'):
        spread_preds = []
        for model_name, model in models['spread'].items():
            pred = model.predict([features['spread']])[0]
            spread_preds.append(pred)

        predictions['spread'] = {
            'prediction': np.mean(spread_preds),
            'range': f"{min(spread_preds):.1f} to {max(spread_preds):.1f}",
            'models': spread_preds
        }

    # Total predictions
    if models.get('total'):
        total_preds = []
        for model_name, model in models['total'].items():
            pred = model.predict([features['total']])[0]
            total_preds.append(pred)

        predictions['total'] = {
            'prediction': np.mean(total_preds),
            'range': f"{min(total_preds):.1f} to {max(total_preds):.1f}",
            'models': total_preds
        }

    # Moneyline predictions
    if models.get('moneyline'):
        moneyline_preds = []
        for model_name, model in models['moneyline'].items():
            pred_proba = model.predict_proba([features['moneyline']])[0]
            home_win_prob = pred_proba[1]  # Probability of home win (class 1)
            moneyline_preds.append(home_win_prob)

        avg_prob = np.mean(moneyline_preds)
        predictions['moneyline'] = {
            'home_win_prob': avg_prob,
            'away_win_prob': 1 - avg_prob,
            'prediction': 'Home' if avg_prob > 0.5 else 'Away',
            'confidence': f"{max(avg_prob, 1-avg_prob):.1%}",
            'models': moneyline_preds
        }

    return predictions

def render_game_prediction(game: Dict, predictions: Dict):
    """Render a game prediction card."""
    st.subheader(f"🏀 {game['away_team']} @ {game['home_team']}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Spread Prediction**")
        if 'spread' in predictions:
            pred = predictions['spread']
            betting_spread = game.get('betting_spread', 'N/A')

            st.metric(
                label=f"Predicted Spread ({betting_spread})",
                value=f"{pred['prediction']:.1f}",
                delta=f"{pred['prediction'] - betting_spread:.1f}" if betting_spread != 'N/A' else None
            )

            if abs(pred['prediction'] - betting_spread) > 3:
                st.success("🎯 Potential Value Bet!")
            elif abs(pred['prediction'] - betting_spread) < 1:
                st.warning("⚠️ Close to line")

    with col2:
        st.markdown("**Total Prediction**")
        if 'total' in predictions:
            pred = predictions['total']
            betting_total = game.get('betting_over_under', 'N/A')

            st.metric(
                label=f"Predicted Total ({betting_total})",
                value=f"{pred['prediction']:.1f}",
                delta=f"{pred['prediction'] - betting_total:.1f}" if betting_total != 'N/A' else None
            )

    with col3:
        st.markdown("**Moneyline Prediction**")
        if 'moneyline' in predictions:
            pred = predictions['moneyline']

            if pred['prediction'] == 'Home':
                st.metric(
                    label=f"{game['home_team']} Win Probability",
                    value=f"{pred['home_win_prob']:.1%}"
                )
            else:
                st.metric(
                    label=f"{game['away_team']} Win Probability",
                    value=f"{pred['away_win_prob']:.1%}"
                )

            st.caption(f"Confidence: {pred['confidence']}")

def main():
    st.set_page_config(
        page_title="March Madness Betting Predictions",
        page_icon="🏀",
        layout="wide"
    )

    st.title("🏀 March Madness Betting Predictions")
    st.markdown("*AI-powered betting predictions using efficiency ratings and team statistics*")

    # Load models
    models = load_models()

    # Sidebar
    st.sidebar.header("Model Performance")
    st.sidebar.metric("Spread MAE", "11.25 pts")
    st.sidebar.metric("Total MAE", "11.58 pts")
    st.sidebar.metric("Moneyline Accuracy", "71.1%")

    # Get upcoming games
    games = get_upcoming_games()

    st.header("🎯 Game Predictions")

    for game in games:
        predictions = make_predictions(game, models)
        render_game_prediction(game, predictions)
        st.divider()

    # Model details
    with st.expander("🤖 Model Details"):
        st.markdown("""
        **Features Used:**
        - Efficiency ratings (offensive, defensive, net)
        - Scoring statistics (PPG, opponent PPG)
        - Four factors (eFG%, turnover rate, ORB%, FTR)
        - Pace and tempo data

        **Models Trained:**
        - Spread: XGBoost, Random Forest, Linear Regression
        - Total: XGBoost, Random Forest, Linear Regression
        - Moneyline: XGBoost, Random Forest, Logistic Regression

        **Training Data:** 2022 NCAA Regular Season (2,194 games)
        """)

if __name__ == "__main__":
    main()