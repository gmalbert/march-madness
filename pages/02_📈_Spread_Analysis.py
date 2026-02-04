"""
Spread Analysis Page
Detailed spread betting analysis with filtering and visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from predictions import (
    load_models,
    get_upcoming_games,
    enrich_espn_game_with_cbbd_data,
    get_team_data,
    make_predictions,
    normalize_team_name,
    sort_games_by_date,
    get_kenpom_barttorvik_data,
    enrich_with_advanced_metrics
)

# Page configuration
st.set_page_config(
    page_title="Spread Analysis - Bracket Oracle",
    page_icon="📈",
    layout="wide"
)

# Logo
logo_path = Path("data_files/logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=250)

st.title("📈 Spread Analysis")
st.markdown("*Detailed spread betting analysis with filtering and visualization*")

# Load models and data
@st.cache_resource
def load_app_data():
    models = load_models()
    efficiency_list, stats_list, season_used = get_team_data()
    kenpom_df, bart_df = get_kenpom_barttorvik_data()
    return models, efficiency_list, stats_list, season_used, kenpom_df, bart_df

with st.spinner("Loading models and team data..."):
    models, efficiency_list, stats_list, season_used, kenpom_df, bart_df = load_app_data()

if not efficiency_list or not stats_list:
    st.error("❌ Could not load team data. Please check data sources.")
    st.stop()

# Get upcoming games
with st.spinner("Loading games..."):
    games = get_upcoming_games()

if not games:
    st.warning("No upcoming games available.")
    st.stop()

st.info(f"Loaded {len(games)} total games")
games = sort_games_by_date(games)

# Filter options
st.sidebar.header("Filter Options")

# Round filter (for tournament games)
round_filter = st.sidebar.selectbox(
    "Tournament Round",
    ["All Games", "First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
)

# Spread range filter
spread_range = st.sidebar.slider(
    "Spread Range",
    min_value=-30.0,
    max_value=30.0,
    value=(-20.0, 20.0),
    step=0.5,
    help="Filter games by spread range"
)

# Minimum edge filter
min_edge = st.sidebar.slider(
    "Minimum Edge (points)",
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.5,
    help="Show only games where predicted margin differs from spread by at least this amount"
)

# Show only ranked matchups
ranked_only = st.sidebar.checkbox("Ranked Teams Only", value=False)

# Process games and generate predictions
spread_picks = []

processing_debug = []
with st.spinner("Analyzing spreads for all games..."):
    for idx, game in enumerate(games):
        # Check if predictions already exist in the game (from flattened precomputed data)
        if 'predicted_margin' in game and game.get('predicted_margin') is not None:
            # Use existing predictions
            enriched_game = game
            predicted_margin = game['predicted_margin']
            spread_pred = {'prediction': predicted_margin, 'confidence': 'Medium'}
        else:
            # Handle both precomputed format (with game_info) and live format (flat)
            if 'game_info' in game:
                # Precomputed format
                game_info = game['game_info']
                enriched_game = {
                    'home_team': game_info.get('home_team'),
                    'away_team': game_info.get('away_team'),
                    'home_eff': game_info.get('home_eff', {}),
                    'away_eff': game_info.get('away_eff', {}),
                    'home_stats': game_info.get('home_stats'),
                    'away_stats': game_info.get('away_stats'),
                    'betting_spread': game_info.get('home_spread'),
                    'betting_over_under': game_info.get('total_line'),
                    'home_moneyline': game_info.get('home_moneyline'),
                    'away_moneyline': game_info.get('away_moneyline'),
                }
            elif 'home_eff' in game and 'away_eff' in game and game.get('home_stats') and game.get('away_stats'):
                # Already enriched (live format) with stats
                enriched_game = game
            else:
                # Need to enrich
                enriched_game = enrich_espn_game_with_cbbd_data(game, efficiency_list, stats_list, season_used)
                if not enriched_game:
                    processing_debug.append(f"Failed to enrich {game.get('away_team', '?')} @ {game.get('home_team', '?')}")
                    continue
            
            # Get advanced metrics
            advanced_metrics = None
            if kenpom_df is not None or bart_df is not None:
                home_team = normalize_team_name(enriched_game['home_team'])
                away_team = normalize_team_name(enriched_game['away_team'])
                advanced_metrics = enrich_with_advanced_metrics(home_team, away_team, kenpom_df, bart_df)
            
            # Generate predictions
            try:
                predictions = make_predictions(enriched_game, models, advanced_metrics)
            except Exception as e:
                continue
            
            # Extract spread prediction
            spread_pred = predictions.get('spread', {})
            if not spread_pred or 'prediction' not in spread_pred:
                continue
            
            predicted_margin = spread_pred['prediction']
        
        # Get actual spread if available (betting_spread is the field from enriched data)
        actual_spread = enriched_game.get('betting_spread')
        if actual_spread is None:
            # Try alternative field names as fallback
            actual_spread = enriched_game.get('spread') or enriched_game.get('home_spread')
        if actual_spread is None:
            processing_debug.append(f"No spread for {enriched_game.get('away_team', '?')} @ {enriched_game.get('home_team', '?')}")
            continue  # Skip if no spread available
        
        # Calculate edge (difference between predicted and actual spread)
        edge = abs(predicted_margin - actual_spread)
        
        # Apply filters
        if ranked_only and not (game.get('home_rank') or game.get('away_rank')):
            continue
        
        if actual_spread < spread_range[0] or actual_spread > spread_range[1]:
            continue
        
        if edge < min_edge:
            continue
        
        # Determine pick
        if predicted_margin < actual_spread:
            pick = enriched_game['away_team']
            pick_type = "AWAY"
        else:
            pick = enriched_game['home_team']
            pick_type = "HOME"
        
        spread_picks.append({
            'matchup': f"{enriched_game['away_team']} @ {enriched_game['home_team']}",
            'away_team': enriched_game['away_team'],
            'home_team': enriched_game['home_team'],
            'spread': actual_spread,
            'predicted_margin': predicted_margin,
            'edge': edge,
            'pick': pick,
            'pick_type': pick_type,
            'away_rank': game.get('away_rank'),
            'home_rank': game.get('home_rank'),
            'confidence': spread_pred.get('confidence', 'Medium')
        })

# Show debugging info
if processing_debug:
    with st.expander("⚠️ Debugging Info: Games Without Spreads", expanded=False):
        st.write(f"{len(processing_debug)} games skipped due to missing spread data:")
        st.info("💡 Betting lines may not be available yet for upcoming games. Run `python fetch_live_odds.py` to fetch current odds, then `python generate_predictions.py` to refresh predictions with betting lines.")
        for msg in processing_debug[:20]:  # Show first 20
            st.caption(msg)

if not spread_picks:
    st.warning("No games match the current filter criteria.")
    st.info(f"**Summary:** Processed {len(games)} games, {len(processing_debug)} had no spread data")
    
    if len(processing_debug) > len(games) * 0.8:  # If more than 80% missing data
        st.error("🔴 Most games are missing betting lines. To fix this:")
        st.code("""# Fetch current odds and regenerate predictions
python fetch_live_odds.py
python generate_predictions.py
""")
    else:
        st.info("Try adjusting the filters (lower minimum edge, wider spread range, disable ranked-only filter)")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(spread_picks)

# Display summary metrics
st.header("📊 Spread Analysis Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Games", len(df))

with col2:
    avg_edge = df['edge'].mean()
    st.metric("Average Edge", f"{avg_edge:.1f} pts")

with col3:
    home_picks = len(df[df['pick_type'] == 'HOME'])
    st.metric("Home Picks", home_picks)

with col4:
    away_picks = len(df[df['pick_type'] == 'AWAY'])
    st.metric("Away Picks", away_picks)

# Display spread picks table
st.header("🎯 Spread Picks")

# Format the display dataframe
display_df = df.copy()
display_df['Matchup'] = display_df['matchup']
display_df['Spread'] = display_df['spread'].apply(lambda x: f"{x:+.1f}")
display_df['Predicted Margin'] = display_df['predicted_margin'].apply(lambda x: f"{x:+.1f}")
display_df['Edge'] = display_df['edge'].apply(lambda x: f"{x:.1f}")
display_df['Pick'] = display_df['pick']
display_df['Confidence'] = display_df['confidence']

st.dataframe(
    display_df[['Matchup', 'Spread', 'Predicted Margin', 'Edge', 'Pick', 'Confidence']],
    use_container_width=True,
    hide_index=True
)

# Visualization: Predicted Margin vs Spread
st.header("📈 Predicted Margin vs Current Spread")

fig = px.scatter(
    df,
    x='spread',
    y='predicted_margin',
    color='pick_type',
    hover_data=['matchup', 'edge', 'confidence'],
    title="Spread Analysis: Predicted Margin vs Current Line",
    labels={
        'spread': 'Current Spread',
        'predicted_margin': 'Predicted Margin',
        'pick_type': 'Pick'
    },
    color_discrete_map={'HOME': '#1f77b4', 'AWAY': '#ff7f0e'}
)

# Add diagonal line (where predicted = actual)
min_val = min(df['spread'].min(), df['predicted_margin'].min())
max_val = max(df['spread'].max(), df['predicted_margin'].max())
fig.add_shape(
    type="line",
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(color="gray", dash="dash", width=2),
    name="Perfect Prediction"
)

# Add zero lines
fig.add_hline(y=0, line_dash="dot", line_color="lightgray", opacity=0.5)
fig.add_vline(x=0, line_dash="dot", line_color="lightgray", opacity=0.5)

fig.update_layout(
    height=600,
    xaxis_title="Current Spread (negative = home favored)",
    yaxis_title="Predicted Margin (negative = away wins by more)",
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

# Edge distribution
st.header("📊 Edge Distribution")

fig2 = px.histogram(
    df,
    x='edge',
    nbins=20,
    title="Distribution of Edges (Prediction vs Spread)",
    labels={'edge': 'Edge (points)', 'count': 'Number of Games'},
    color_discrete_sequence=['#636EFA']
)

fig2.update_layout(
    height=400,
    showlegend=False
)

st.plotly_chart(fig2, use_container_width=True)

# Top picks by edge
st.header("⭐ Top Picks by Edge")

top_picks = df.nlargest(10, 'edge')

for idx, row in top_picks.iterrows():
    with st.container():
        col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 2])
        
        with col1:
            st.write(f"**{row['matchup']}**")
        
        with col2:
            st.metric("Spread", f"{row['spread']:+.1f}")
        
        with col3:
            st.metric("Predicted", f"{row['predicted_margin']:+.1f}")
        
        with col4:
            st.metric("Edge", f"{row['edge']:.1f}")
        
        with col5:
            st.success(f"**Pick: {row['pick']}**")
        
        st.divider()

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Games above the diagonal line favor the away team, below favor the home team. Larger distances from the line indicate stronger betting opportunities.")
