import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
from typing import Dict, List, Optional
import sys
import os
from datetime import datetime, timezone
from dateutil import parser as date_parser
import pytz

# Add the current directory to the path so we can import data_collection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_collection import fetch_games, fetch_betting_lines, fetch_adjusted_efficiency, fetch_team_stats
except ImportError:
    st.error("Could not import data collection functions. Make sure data_collection.py is in the same directory.")
    st.stop()

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

def format_game_datetime(game) -> str:
    """Format game date/time in Eastern Time."""
    try:
        # Get the start date from the game
        if isinstance(game, dict):
            start_date_str = game.get('startDate') or game.get('start_date')
        else:
            start_date_str = getattr(game, 'startDate', None) or getattr(game, 'start_date', None)

        if not start_date_str:
            return "Date TBD"

        # Parse the date (it's in ISO format with timezone)
        if isinstance(start_date_str, str):
            game_datetime = date_parser.parse(start_date_str)
        else:
            game_datetime = start_date_str

        # Convert to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        game_datetime_et = game_datetime.astimezone(eastern)

        # Format nicely
        now = datetime.now(eastern)
        if game_datetime_et.date() == now.date():
            # Today
            return f"Today {game_datetime_et.strftime('%I:%M %p ET')}"
        elif (game_datetime_et - now).days == 1:
            # Tomorrow
            return f"Tomorrow {game_datetime_et.strftime('%I:%M %p ET')}"
        elif game_datetime_et.year == now.year:
            # This year
            return game_datetime_et.strftime('%b %d, %I:%M %p ET')
        else:
            # Different year
            return game_datetime_et.strftime('%b %d, %Y %I:%M %p ET')

    except Exception as e:
        return "Date TBD"

def sort_games_by_date(games: List) -> List:
    """Sort games by start date (most recent first)."""
    def get_game_datetime(game):
        try:
            if isinstance(game, dict):
                start_date_str = game.get('startDate') or game.get('start_date')
            else:
                start_date_str = getattr(game, 'startDate', None) or getattr(game, 'start_date', None)

            if start_date_str:
                return date_parser.parse(start_date_str)
            else:
                # If no date, put at the end
                return datetime.min.replace(tzinfo=timezone.utc)
        except:
            return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(games, key=get_game_datetime, reverse=True)

def get_upcoming_games() -> List[Dict]:
    """Get games for analysis - shows recent games since upcoming games may not be scheduled yet."""
    try:
        # For January 2026, we're in the 2025-2026 season
        current_year = 2026  # API uses the end year of the season

        # Try to fetch both postseason and regular season games
        all_games = []
        postseason_games = fetch_games(current_year, "postseason")
        regular_games = fetch_games(current_year, "regular")

        all_games.extend(postseason_games)
        all_games.extend(regular_games)

        st.info(f"Found {len(postseason_games)} postseason games and {len(regular_games)} regular season games")

        if not all_games:
            st.warning("No games found for the current season. Using sample data.")
            return get_sample_games()

        # Check for upcoming games
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        upcoming_games = []

        for game in all_games:
            if isinstance(game, dict):
                start_date_str = game.get('startDate') or game.get('start_date')
            else:
                start_date_str = getattr(game, 'startDate', None) or getattr(game, 'start_date', None)

            if start_date_str:
                try:
                    game_datetime = date_parser.parse(start_date_str)
                    game_datetime_et = game_datetime.astimezone(eastern)
                    if game_datetime_et > now:
                        upcoming_games.append(game)
                except:
                    pass

        if upcoming_games:
            st.success(f"Found {len(upcoming_games)} upcoming games!")
            # Sort upcoming games by date
            games_to_show = sort_games_by_date(upcoming_games)[:20]
        else:
            st.info("No upcoming games found. This may be because:")
            st.info("• College basketball season may be between conference tournaments")
            st.info("• March Madness hasn't started yet (typically March)")
            st.info("• Showing recent games for analysis instead")
            # Since we're in January 2026, most games are already played
            # Show recent games (last 20) for analysis
            games_to_show = all_games[-20:] if len(all_games) >= 20 else all_games

        # Convert API game objects to our expected format
        formatted_games = []
        for game in games_to_show:
            try:
                formatted_game = format_game_data(game)
                if formatted_game:
                    # Include the original game object for date access
                    formatted_game['_original_game'] = game
                    formatted_games.append(formatted_game)
                else:
                    pass  # Silently skip games that can't be formatted
            except Exception as e:
                pass  # Silently skip games with formatting errors

        st.info(f"Successfully formatted {len(formatted_games)} games")

        return formatted_games if formatted_games else get_sample_games()

    except Exception as e:
        st.error(f"Error fetching games: {e}")
        st.info("Falling back to sample data.")
        return get_sample_games()

def get_sample_games() -> List[Dict]:
    """Return sample games for demo purposes."""
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

def format_game_data(game) -> Optional[Dict]:
    """Convert API game object to our expected format."""
    try:
        # Extract team names - handle both dict and object formats
        if isinstance(game, dict):
            # Dict format from API
            home_team = game.get('homeTeam') or game.get('home_team')
            away_team = game.get('awayTeam') or game.get('away_team')
        else:
            # Object format
            home_team = (getattr(game, 'home_team', None) or
                        getattr(game, 'home', None) or
                        getattr(game, 'homeTeam', None))
            away_team = (getattr(game, 'away_team', None) or
                        getattr(game, 'away', None) or
                        getattr(game, 'awayTeam', None))

        if not home_team or not away_team:
            return None

        # Get team names as strings
        if isinstance(home_team, str):
            home_team_name = home_team
        elif hasattr(home_team, 'name'):
            home_team_name = home_team.name
        else:
            home_team_name = str(home_team)

        if isinstance(away_team, str):
            away_team_name = away_team
        elif hasattr(away_team, 'name'):
            away_team_name = away_team.name
        else:
            away_team_name = str(away_team)

        # Try to get efficiency data for both teams
        try:
            efficiency_data = fetch_adjusted_efficiency(2026)  # Current season
            home_eff = next((e for e in efficiency_data if getattr(e, 'team', None) == home_team_name), {})
            away_eff = next((e for e in efficiency_data if getattr(e, 'team', None) == away_team_name), {})
        except:
            home_eff = {}
            away_eff = {}

        # Try to get team stats
        try:
            team_stats_data = fetch_team_stats(2026)
            home_stats = next((s for s in team_stats_data if getattr(s, 'team', None) == home_team_name), {})
            away_stats = next((s for s in team_stats_data if getattr(s, 'team', None) == away_team_name), {})
        except:
            home_stats = {}
            away_stats = {}

        # Try to get betting lines
        betting_spread = None
        betting_over_under = None
        try:
            lines_data = fetch_betting_lines(2026, "postseason")
            # Find lines for this specific game - try multiple matching criteria
            for line in lines_data:
                if isinstance(line, dict):
                    line_home = line.get('homeTeam') or line.get('home_team')
                    line_away = line.get('awayTeam') or line.get('away_team')
                else:
                    line_home = (getattr(line, 'home_team', None) or
                               getattr(line, 'homeTeam', None) or
                               getattr(line, 'home', None))
                    line_away = (getattr(line, 'away_team', None) or
                               getattr(line, 'awayTeam', None) or
                               getattr(line, 'away', None))

                if ((line_home == home_team_name and line_away == away_team_name) or
                    (line_home == home_team and line_away == away_team)):
                    betting_spread = getattr(line, 'spread', None)
                    betting_over_under = getattr(line, 'over_under', None)
                    break
        except:
            pass

        # Get actual scores if available
        if isinstance(game, dict):
            actual_home_score = game.get('homePoints') or game.get('home_points')
            actual_away_score = game.get('awayPoints') or game.get('away_points')
        else:
            actual_home_score = getattr(game, 'home_points', None) or getattr(game, 'homePoints', None)
            actual_away_score = getattr(game, 'away_points', None) or getattr(game, 'awayPoints', None)

        # Create the game data structure with default values if data is missing
        return {
            "home_team": home_team_name,
            "away_team": away_team_name,
            "actual_home_score": actual_home_score,
            "actual_away_score": actual_away_score,
            "home_eff": {
                "offensiveRating": float(getattr(home_eff, 'offensive_rating', 100) or getattr(home_eff, 'adj_offense', 100) or 100),
                "defensiveRating": float(getattr(home_eff, 'defensive_rating', 100) or getattr(home_eff, 'adj_defense', 100) or 100),
                "netRating": float(getattr(home_eff, 'net_rating', 0) or getattr(home_eff, 'adj_net', 0) or 0)
            },
            "away_eff": {
                "offensiveRating": float(getattr(away_eff, 'offensive_rating', 100) or getattr(away_eff, 'adj_offense', 100) or 100),
                "defensiveRating": float(getattr(away_eff, 'defensive_rating', 100) or getattr(away_eff, 'adj_defense', 100) or 100),
                "netRating": float(getattr(away_eff, 'net_rating', 0) or getattr(away_eff, 'adj_net', 0) or 0)
            },
            "home_stats": {
                "ppg": float(getattr(home_stats, 'points_per_game', 70) or getattr(home_stats, 'ppg', 70) or 70),
                "pace": float(getattr(home_stats, 'pace', 68) or 68),
                "efg_pct": float(getattr(home_stats, 'effective_fg_pct', 0.50) or getattr(home_stats, 'efg_pct', 0.50) or 0.50),
                "to_rate": float(getattr(home_stats, 'turnover_pct', 0.15) or getattr(home_stats, 'to_rate', 0.15) or 0.15),
                "orb_pct": float(getattr(home_stats, 'orb_pct', 0.30) or 0.30),
                "ft_rate": float(getattr(home_stats, 'ft_rate', 0.35) or 0.35),
                "opp_ppg": float(getattr(home_stats, 'opp_points_per_game', 70) or getattr(home_stats, 'opp_ppg', 70) or 70)
            },
            "away_stats": {
                "ppg": float(getattr(away_stats, 'points_per_game', 70) or getattr(away_stats, 'ppg', 70) or 70),
                "pace": float(getattr(away_stats, 'pace', 68) or 68),
                "efg_pct": float(getattr(away_stats, 'effective_fg_pct', 0.50) or getattr(away_stats, 'efg_pct', 0.50) or 0.50),
                "to_rate": float(getattr(away_stats, 'turnover_pct', 0.15) or getattr(away_stats, 'to_rate', 0.15) or 0.15),
                "orb_pct": float(getattr(away_stats, 'orb_pct', 0.30) or 0.30),
                "ft_rate": float(getattr(away_stats, 'ft_rate', 0.35) or 0.35),
                "opp_ppg": float(getattr(away_stats, 'opp_points_per_game', 70) or getattr(away_stats, 'opp_ppg', 70) or 70)
            },
            "betting_spread": betting_spread,
            "betting_over_under": betting_over_under
        }

    except Exception as e:
        print(f"Error formatting game {game}: {e}")
        return None

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

    # Show actual score if available
    actual_home = game.get('actual_home_score')
    actual_away = game.get('actual_away_score')

    if actual_home is not None and actual_away is not None:
        actual_spread = actual_home - actual_away
        actual_total = actual_home + actual_away

        st.markdown(f"**Final Score:** {game['away_team']} {actual_away} - {game['home_team']} {actual_home}")
        st.markdown(f"**Actual Spread:** {actual_spread:+.0f} | **Actual Total:** {actual_total:.0f}")
        st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Spread Prediction**")
        if 'spread' in predictions:
            pred = predictions['spread']
            betting_spread = game.get('betting_spread', 'N/A')

            st.metric(
                label=f"Predicted Spread ({betting_spread if betting_spread is not None else 'N/A'})",
                value=f"{pred['prediction']:.1f}",
                delta=f"{pred['prediction'] - betting_spread:.1f}" if betting_spread is not None and isinstance(betting_spread, (int, float)) else None
            )

            if actual_home is not None and actual_away is not None:
                actual_spread = actual_home - actual_away
                spread_diff = pred['prediction'] - actual_spread
                if abs(spread_diff) <= 3:
                    st.success(f"✅ Accurate! (Off by {abs(spread_diff):.1f} pts)")
                elif abs(spread_diff) <= 7:
                    st.warning(f"⚠️ Close (Off by {abs(spread_diff):.1f} pts)")
                else:
                    st.error(f"❌ Missed (Off by {abs(spread_diff):.1f} pts)")

            if betting_spread is not None and isinstance(betting_spread, (int, float)) and abs(pred['prediction'] - betting_spread) > 3:
                st.success("🎯 Potential Value Bet!")
            elif betting_spread is not None and isinstance(betting_spread, (int, float)) and abs(pred['prediction'] - betting_spread) < 1:
                st.warning("⚠️ Close to line")
                st.warning("⚠️ Close to line")

    with col2:
        st.markdown("**Total Prediction**")
        if 'total' in predictions:
            pred = predictions['total']
            betting_total = game.get('betting_over_under', 'N/A')

            st.metric(
                label=f"Predicted Total ({betting_total if betting_total is not None else 'N/A'})",
                value=f"{pred['prediction']:.1f}",
                delta=f"{pred['prediction'] - betting_total:.1f}" if betting_total is not None and isinstance(betting_total, (int, float)) else None
            )

            if actual_home is not None and actual_away is not None:
                actual_total = actual_home + actual_away
                total_diff = pred['prediction'] - actual_total
                if abs(total_diff) <= 5:
                    st.success(f"✅ Accurate! (Off by {abs(total_diff):.1f} pts)")
                elif abs(total_diff) <= 10:
                    st.warning(f"⚠️ Close (Off by {abs(total_diff):.1f} pts)")
                else:
                    st.error(f"❌ Missed (Off by {abs(total_diff):.1f} pts)")

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

            if actual_home is not None and actual_away is not None:
                actual_winner = 'Home' if actual_home > actual_away else 'Away'
                if pred['prediction'] == actual_winner:
                    st.success("✅ Correct winner prediction!")
                else:
                    st.error("❌ Wrong winner prediction")

            st.caption(f"Confidence: {pred['confidence']}")

def main():
    st.set_page_config(
        page_title="Bracket Oracle - March Madness Predictions",
        page_icon="🏀",
        layout="wide"
    )

    # Logo
    st.image("data_files/logo.png", width=250)

    st.title("🏀 Bracket Oracle - March Madness Predictions")
    st.markdown("*AI-powered betting predictions using efficiency ratings and team statistics*")

    # Load models
    models = load_models()

    # Sidebar
    st.sidebar.header("Model Performance")
    st.sidebar.metric("Spread MAE", "11.25 pts")
    st.sidebar.metric("Total MAE", "11.58 pts")
    st.sidebar.metric("Moneyline Accuracy", "71.1%")

    st.sidebar.header("Season Status")
    st.sidebar.info("January 2026 - Regular season completed. March Madness begins in March.")

    # Get upcoming games
    games = get_upcoming_games()

    if not games:
        st.error("No games available. Please check your API connection and try again.")
        return

    # Sort games by date (most recent first)
    games = sort_games_by_date(games)

    st.header("🎯 Game Analysis")
    st.markdown("*Analyzing recent games - no upcoming games currently scheduled*")

    # Game selector with date/time
    game_options = []
    for game in games:
        # Get the date/time from the original game object
        original_game = game.get('_original_game')
        date_str = format_game_datetime(original_game) if original_game else "Date TBD"
        game_options.append(f"{game['away_team']} @ {game['home_team']} - {date_str}")

    selected_game_idx = st.selectbox(
        "Select a game to analyze:",
        range(len(game_options)),
        format_func=lambda x: game_options[x]
    )

    selected_game = games[selected_game_idx]

    # Make predictions for selected game
    predictions = make_predictions(selected_game, models)

    # Display the prediction
    render_game_prediction(selected_game, predictions)

    st.info("💡 **Note:** Showing recent games for analysis. Predictions show what the model would have forecasted before the game.")

    # Show all games in an expander
    with st.expander("📅 View All Available Games"):
        for i, game in enumerate(games):
            original_game = game.get('_original_game')
            date_str = format_game_datetime(original_game) if original_game else "Date TBD"
            game_display = f"{game['away_team']} @ {game['home_team']} - {date_str}"

            if i == selected_game_idx:
                st.markdown(f"**→ {game_display}** (currently selected)")
            else:
                st.markdown(game_display)
            if i < len(games) - 1:
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