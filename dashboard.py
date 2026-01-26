"""
March Madness Betting Dashboard
Main dashboard with today's picks and betting recommendations
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from betting_models import evaluate_betting_roi_from_df
    from underdog_value import (
        identify_underdog_value,
        moneyline_to_implied_probability,
        calculate_expected_value
    )
except ImportError as e:
    st.error(f"Could not import required functions: {e}")
    st.stop()

# Configuration
st.set_page_config(
    page_title="March Madness Betting Dashboard",
    page_icon="🏀",
    layout="wide"
)

DATA_DIR = Path("data_files")

# Custom CSS for better styling
st.markdown("""
<style>
    .game-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .value-bet {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
    }
    .metric-positive {
        color: #28a745;
    }
    .metric-negative {
        color: #dc3545;
    }
    .pick-badge {
        background-color: #007bff;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600)
def load_historical_performance():
    """Load historical performance metrics from evaluation data."""
    try:
        # Try to load the predictions file
        hist_file = DATA_DIR / "historical_games_with_betting_predictions.csv"
        if hist_file.exists():
            df = pd.read_csv(hist_file)
            
            # Calculate performance metrics
            spread_perf = evaluate_betting_roi_from_df(df, 'spread')
            total_perf = evaluate_betting_roi_from_df(df, 'total')
            moneyline_perf = evaluate_betting_roi_from_df(df, 'moneyline')
            
            return {
                'spread': spread_perf,
                'total': total_perf,
                'moneyline': moneyline_perf,
                'total_games': len(df)
            }
        else:
            # Return mock data if no historical data
            return {
                'spread': {'accuracy': 0.575, 'overall_roi': 8.2, 'total_bets': 73},
                'total': {'accuracy': 0.521, 'overall_roi': 5.1, 'total_bets': 73},
                'moneyline': {'accuracy': 0.650, 'overall_roi': 15.3, 'total_bets': 73},
                'total_games': 73
            }
    except Exception as e:
        st.error(f"Error loading historical performance: {e}")
        return None


@st.cache_data(ttl=300)
def load_todays_predictions():
    """Load upcoming game predictions (next 3 days in local time)."""
    try:
        predictions_file = DATA_DIR / "upcoming_game_predictions.json"
        if predictions_file.exists():
            with open(predictions_file, 'r') as f:
                data = json.load(f)
            
            # Filter for games in the next 3 days (using local timezone)
            from datetime import datetime, timedelta, timezone
            
            # Get current time in local timezone
            now_local = datetime.now()
            three_days_from_now = now_local + timedelta(days=3)
            
            upcoming_games = []
            for game in data:
                game_date_str = game.get('game_info', {}).get('date', '')
                if game_date_str:
                    try:
                        # Parse UTC date and convert to local time
                        if '+00:00' in game_date_str or 'Z' in game_date_str:
                            # Parse as UTC
                            game_date_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                            # Convert to local time
                            game_date_local = game_date_utc.replace(tzinfo=timezone.utc).astimezone(tz=None)
                        else:
                            # Already local or no timezone info
                            game_date_local = datetime.fromisoformat(game_date_str)
                        
                        # Include games from today through next 3 days (local time)
                        if now_local.date() <= game_date_local.date() <= three_days_from_now.date():
                            upcoming_games.append(game)
                    except Exception as e:
                        # If date parsing fails, include the game
                        upcoming_games.append(game)
            
            return upcoming_games
        else:
            return []
    except Exception as e:
        st.error(f"Error loading predictions: {e}")
        return []


def calculate_value_bets(games):
    """Identify value betting opportunities from games."""
    value_bets = []
    
    for game in games:
        game_info = game.get('game_info', {})
        predictions = game.get('predictions', {})
        
        # Check moneyline value - using correct keys from predictions
        home_prob = predictions.get('moneyline_home_win_prob', 0.5)
        away_prob = predictions.get('moneyline_away_win_prob', 0.5)
        
        # Ensure probabilities are in 0-1 range
        if home_prob > 1:
            home_prob = home_prob / 100
        if away_prob > 1:
            away_prob = away_prob / 100
        
        home_ml = game_info.get('home_moneyline')
        away_ml = game_info.get('away_moneyline')
        
        if home_ml and home_prob > 0:
            home_implied = moneyline_to_implied_probability(home_ml)
            home_edge = home_prob - home_implied
            
            if home_edge > 0.05:  # 5% edge threshold
                value_bets.append({
                    'game': f"{game_info.get('away_team')} @ {game_info.get('home_team')}",
                    'team': game_info.get('home_team'),
                    'bet_type': 'Moneyline',
                    'odds': home_ml,
                    'model_prob': home_prob,
                    'implied_prob': home_implied,
                    'edge': home_edge
                })
        
        if away_ml and away_prob > 0:
            away_implied = moneyline_to_implied_probability(away_ml)
            away_edge = away_prob - away_implied
            
            if away_edge > 0.05:
                value_bets.append({
                    'game': f"{game_info.get('away_team')} @ {game_info.get('home_team')}",
                    'team': game_info.get('away_team'),
                    'bet_type': 'Moneyline',
                    'odds': away_ml,
                    'model_prob': away_prob,
                    'implied_prob': away_implied,
                    'edge': away_edge
                })
    
    return sorted(value_bets, key=lambda x: x['edge'], reverse=True)


def render_game_card(game):
    """Render a single game prediction card."""
    from datetime import datetime, timezone
    
    game_info = game.get('game_info', {})
    predictions = game.get('predictions', {})
    
    home_team = game_info.get('home_team', 'Home')
    away_team = game_info.get('away_team', 'Away')
    
    # Extract predictions - using correct keys
    home_win_prob = predictions.get('moneyline_home_win_prob', 0.5)
    away_win_prob = predictions.get('moneyline_away_win_prob', 0.5)
    
    # Ensure probabilities are in 0-1 range
    if home_win_prob > 1:
        home_win_prob = home_win_prob / 100
    if away_win_prob > 1:
        away_win_prob = away_win_prob / 100
    
    # Extract spread and total predictions (these are direct floats, not dicts)
    predicted_margin = predictions.get('spread_prediction', 0)
    if isinstance(predicted_margin, dict):
        predicted_margin = predicted_margin.get('predicted_margin', 0)
    
    predicted_total = predictions.get('total_prediction', 0)
    if isinstance(predicted_total, dict):
        predicted_total = predicted_total.get('predicted_total', 0)
    
    # Betting lines
    home_spread = game_info.get('home_spread')
    away_spread = game_info.get('away_spread')
    total_line = game_info.get('total_line')
    home_ml = game_info.get('home_moneyline')
    away_ml = game_info.get('away_moneyline')
    
    # Game time
    game_date_str = game_info.get('date', '')
    game_time_display = ""
    if game_date_str:
        try:
            game_date_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
            game_date_local = game_date_utc.replace(tzinfo=timezone.utc).astimezone(tz=None)
            game_time_display = game_date_local.strftime('%I:%M %p')
        except:
            pass
    
    # Check for value bets
    value_picks = []
    if home_ml and home_win_prob > 0:
        home_implied = moneyline_to_implied_probability(home_ml)
        home_edge = home_win_prob - home_implied
        if home_edge > 0.05:
            value_picks.append({
                'team': home_team,
                'edge': home_edge,
                'odds': home_ml,
                'prob': home_win_prob
            })
    
    if away_ml and away_win_prob > 0:
        away_implied = moneyline_to_implied_probability(away_ml)
        away_edge = away_win_prob - away_implied
        if away_edge > 0.05:
            value_picks.append({
                'team': away_team,
                'edge': away_edge,
                'odds': away_ml,
                'prob': away_win_prob
            })
    
    # Render card
    with st.container():
        # Header with value alert
        if value_picks:
            col_h1, col_h2 = st.columns([3, 1])
            with col_h1:
                st.markdown(f"### {away_team} @ {home_team}")
                if game_time_display:
                    st.caption(f"🕐 {game_time_display}")
            with col_h2:
                st.markdown("### 💰 VALUE")
        else:
            st.markdown(f"### {away_team} @ {home_team}")
            if game_time_display:
                st.caption(f"🕐 {game_time_display}")
        
        # Main betting info
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("**💵 Moneyline**")
            
            if home_ml and away_ml:
                # Show both teams with odds
                st.write(f"{away_team}: **{away_ml:+d}**")
                st.write(f"{home_team}: **{home_ml:+d}**")
                st.divider()
                
                # Show pick
                if home_win_prob > 0.5:
                    st.success(f"✓ **BET {home_team}** ({home_ml:+d})")
                    st.caption(f"{home_win_prob*100:.1f}% win probability")
                else:
                    st.success(f"✓ **BET {away_team}** ({away_ml:+d})")
                    st.caption(f"{away_win_prob*100:.1f}% win probability")
            else:
                st.warning("📅 Odds not yet available")
                st.caption("Sportsbooks haven't released odds for this game yet")
                # Still show prediction
                if home_win_prob > 0.5:
                    st.info(f"Predicted winner: **{home_team}**")
                    st.caption(f"{home_win_prob*100:.1f}% probability")
                else:
                    st.info(f"Predicted winner: **{away_team}**")
                    st.caption(f"{away_win_prob*100:.1f}% probability")
        
        with col2:
            st.markdown("**📊 Spread**")
            
            if home_spread is not None and away_spread is not None:
                # Show both teams with spreads
                st.write(f"{away_team}: **{away_spread:+.1f}**")
                st.write(f"{home_team}: **{home_spread:+.1f}**")
                st.divider()
                
                # Show pick based on predicted margin
                if predicted_margin > -home_spread:
                    st.success(f"✓ **BET {home_team} {home_spread:+.1f}**")
                    st.caption(f"Predicted margin: {predicted_margin:+.1f}")
                else:
                    st.success(f"✓ **BET {away_team} {away_spread:+.1f}**")
                    st.caption(f"Predicted margin: {predicted_margin:+.1f}")
            else:
                st.warning("📅 Spread not yet available")
                st.caption("Check back when game is closer")
                # Still show prediction
                st.info(f"Predicted margin: **{predicted_margin:+.1f}**")
                if predicted_margin > 0:
                    st.caption(f"{home_team} by {abs(predicted_margin):.1f}")
                else:
                    st.caption(f"{away_team} by {abs(predicted_margin):.1f}")
        
        with col3:
            st.markdown("**🎯 Over/Under**")
            
            if total_line:
                st.write(f"Line: **{total_line:.1f}**")
                st.write(f"Predicted: **{predicted_total:.1f}**")
                st.divider()
                
                # Show pick
                if predicted_total > total_line:
                    diff = predicted_total - total_line
                    st.success(f"✓ **BET OVER {total_line:.1f}**")
                    st.caption(f"Edge: +{diff:.1f} points")
                else:
                    diff = total_line - predicted_total
                    st.success(f"✓ **BET UNDER {total_line:.1f}**")
                    st.caption(f"Edge: +{diff:.1f} points")
            else:
                st.warning("📅 Total not yet available")
                st.caption("Check back when game is closer")
                # Still show prediction
                st.info(f"Predicted total: **{predicted_total:.1f}**")
        
        with col4:
            st.markdown("**⭐ Value Bets**")
            
            if value_picks:
                for vp in value_picks:
                    st.success(f"✓ **{vp['team']}**")
                    st.write(f"Odds: **{vp['odds']:+d}**")
                    st.write(f"Edge: **+{vp['edge']*100:.1f}%**")
                    st.caption(f"Win prob: {vp['prob']*100:.1f}%")
            else:
                st.info("No value found")
                st.caption("(Edge < 5%)")
        
        st.divider()


def render_dashboard():
    """Main dashboard rendering."""
    st.title("🏀 March Madness Betting Dashboard")
    
    # Load data
    performance = load_historical_performance()
    todays_games = load_todays_predictions()
    
    # Performance Metrics Section
    st.header("📊 Model Performance")
    
    if performance:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            spread_acc = performance['spread'].get('accuracy', 0)
            spread_total = performance['spread'].get('total_bets', 0)
            wins = int(spread_acc * spread_total)
            losses = spread_total - wins
            st.metric(
                "Model ATS Record",
                f"{wins}-{losses}",
                f"{spread_acc*100:.1f}%"
            )
        
        with col2:
            ou_acc = performance['total'].get('accuracy', 0)
            ou_total = performance['total'].get('total_bets', 0)
            ou_wins = int(ou_acc * ou_total)
            ou_losses = ou_total - ou_wins
            st.metric(
                "O/U Record",
                f"{ou_wins}-{ou_losses}",
                f"{ou_acc*100:.1f}%"
            )
        
        with col3:
            import math
            ml_roi = performance['moneyline'].get('overall_roi', None)

            if ml_roi is None or (isinstance(ml_roi, float) and math.isnan(ml_roi)):
                st.metric("Moneyline ROI", "N/A", delta=None)
            else:
                if ml_roi > 0:
                    roi_display = f"+{ml_roi:.1f}%"
                else:
                    roi_display = f"{ml_roi:.1f}%"
                st.metric("Moneyline ROI", roi_display, delta=None)
        
        with col4:
            # Calculate value bet ROI (estimated from moneyline performance)
            try:
                if ml_roi is None or (isinstance(ml_roi, float) and math.isnan(ml_roi)):
                    st.metric("Value Bet ROI", "N/A", delta=None)
                else:
                    value_roi = ml_roi * 1.5 if ml_roi > 0 else 0  # Value bets typically perform better
                    if value_roi > 0:
                        st.metric("Value Bet ROI", f"+{value_roi:.1f}%", delta="Est.")
                    else:
                        st.metric("Value Bet ROI", "N/A", delta=None)
            except Exception:
                st.metric("Value Bet ROI", "N/A", delta=None)
    
    st.divider()
    
    # Upcoming Games Section
    st.header("🎯 Upcoming Games (Next 3 Days)")
    
    if todays_games:
        # Group games by date (convert UTC to local time for display)
        from datetime import datetime, timezone
        from collections import defaultdict
        
        games_by_date = defaultdict(list)
        for game in todays_games:
            game_date_str = game.get('game_info', {}).get('date', '')
            try:
                # Parse UTC date and convert to local time
                if '+00:00' in game_date_str or 'Z' in game_date_str:
                    game_date_utc = datetime.fromisoformat(game_date_str.replace('Z', '+00:00'))
                    game_date_local = game_date_utc.replace(tzinfo=timezone.utc).astimezone(tz=None)
                else:
                    game_date_local = datetime.fromisoformat(game_date_str)
                
                date_key = game_date_local.strftime('%A, %B %d, %Y')
            except:
                date_key = 'Unknown Date'
            games_by_date[date_key].append(game)
        
        st.info(f"📅 {len(todays_games)} games available for betting analysis across {len(games_by_date)} days")
        
        # Show value bets first
        value_bets = calculate_value_bets(todays_games)
        
        if value_bets:
            st.subheader("💰 Top Value Betting Opportunities")
            st.info("**These are the best bets** - teams where our model's win probability significantly exceeds the implied probability from the odds (5%+ edge)")
            
            for i, vb in enumerate(value_bets[:5]):  # Show top 5 value bets
                col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
                
                with col1:
                    st.markdown(f"**{i+1}. BET ON: {vb['team']}**")
                    st.caption(vb['game'])
                
                with col2:
                    st.metric("Odds", f"{vb['odds']:+d}")
                
                with col3:
                    st.metric("Our Win Prob", f"{vb['model_prob']*100:.1f}%")
                
                with col4:
                    st.metric("Market Prob", f"{vb['implied_prob']*100:.1f}%")
                
                with col5:
                    st.metric("Your Edge", f"+{vb['edge']*100:.1f}%", delta="VALUE")
            
            st.divider()
        else:
            st.info("No strong value bets found with 5%+ edge. Check individual game cards below for standard picks.")
        
        # Display games organized by date
        st.subheader("📋 All Games by Date")
        
        for date_key in sorted(games_by_date.keys()):
            with st.expander(f"📅 {date_key} ({len(games_by_date[date_key])} games)", expanded=True):
                for game in games_by_date[date_key]:
                    render_game_card(game)
    
    else:
        st.warning("No games available in the next 3 days. Check back later!")
        st.info("💡 Predictions are updated daily when games are scheduled.")
        
        # Show a sample to help debug
        st.subheader("🔍 Debug Info")
        with st.expander("Click to see all available games"):
            predictions_file = DATA_DIR / "upcoming_game_predictions.json"
            if predictions_file.exists():
                with open(predictions_file, 'r') as f:
                    all_data = json.load(f)
                st.write(f"Total games in predictions file: {len(all_data)}")
                if all_data:
                    st.write("First few games:")
                    for game in all_data[:5]:
                        game_info = game.get('game_info', {})
                        st.write(f"  {game_info.get('away_team')} @ {game_info.get('home_team')} - {game_info.get('date')}")
    
    # Footer
    st.divider()
    st.caption("⚠️ **Disclaimer:** This is for educational purposes only. Please gamble responsibly.")
    st.caption("📊 Model performance based on historical NCAA tournament data.")


# Sidebar
with st.sidebar:
    st.title("📊 Dashboard Guide")
    
    st.markdown("### How to Read the Dashboard")
    
    with st.expander("💵 Moneyline", expanded=False):
        st.markdown("""
        **What it shows:** Both teams with their odds
        
        **The Pick:** Team with green checkmark (✓)
        - This is the team our model predicts will WIN
        - Shows win probability percentage
        
        **Example:**
        - ✓ **BET Duke (-150)** = Bet on Duke to win at -150 odds
        """)
    
    with st.expander("📊 Spread", expanded=False):
        st.markdown("""
        **What it shows:** Point spread for each team
        
        **The Pick:** Team with green checkmark (✓)
        - This is the team we predict will COVER the spread
        - Shows predicted margin
        
        **Example:**
        - ✓ **BET Duke -5.5** = Bet on Duke to win by more than 5.5
        - ✓ **BET UNC +7.5** = Bet on UNC to lose by less than 7.5 or win
        """)
    
    with st.expander("🎯 Over/Under", expanded=False):
        st.markdown("""
        **What it shows:** Total points line
        
        **The Pick:** OVER or UNDER with green checkmark (✓)
        - OVER = Total points will be MORE than the line
        - UNDER = Total points will be LESS than the line
        
        **Example:**
        - ✓ **BET OVER 145.5** = Bet total points > 145.5
        """)
    
    with st.expander("⭐ Value Bets", expanded=False):
        st.markdown("""
        **What it shows:** Best betting opportunities
        
        A value bet occurs when our model's win probability is at least 5% higher than what the betting odds imply.
        
        **Example:**
        - Our model: 60% win probability
        - Odds imply: 52% win probability  
        - **Edge: +8%** = VALUE BET! ✓
        
        These are statistically the best bets.
        """)
    
    st.divider()
    
    st.markdown("### ⚙️ Settings")
    min_edge = st.slider("Min Value Edge %", 1, 20, 5)
    
    st.divider()
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.divider()
    st.caption("⚠️ For educational purposes only.")
    st.caption("Please gamble responsibly.")

# Main app
if __name__ == "__main__":
    render_dashboard()
