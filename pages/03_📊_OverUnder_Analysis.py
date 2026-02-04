"""
Over/Under Analysis Page
Detailed over/under betting analysis with tempo filtering
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
    page_title="Over/Under Analysis - Bracket Oracle",
    page_icon="📊",
    layout="wide"
)

# Logo
logo_path = Path("data_files/logo.png")
if logo_path.exists():
    st.image(str(logo_path), width=250)

st.title("📊 Over/Under Analysis")
st.markdown("*Detailed over/under betting analysis with pace and tempo filtering*")

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

# Pace/tempo filter
tempo_filter = st.sidebar.selectbox(
    "Pace Profile",
    ["All Games", "High Tempo (72+ possessions)", "Medium Tempo (68-72)", "Low Tempo (<68)", "Pace Mismatch (5+ difference)"],
    help="Filter games by team pace/tempo"
)

# Total range filter
total_range = st.sidebar.slider(
    "Total Range",
    min_value=100.0,
    max_value=200.0,
    value=(130.0, 180.0),
    step=1.0,
    help="Filter games by over/under line"
)

# Minimum edge filter
min_edge = st.sidebar.slider(
    "Minimum Edge (points)",
    min_value=0.0,
    max_value=15.0,
    value=0.0,
    step=0.5,
    help="Show only games where predicted total differs from line by at least this amount"
)

# Show only ranked matchups
ranked_only = st.sidebar.checkbox("Ranked Teams Only", value=False)

# Process games and generate predictions
ou_picks = []

processing_debug = []
with st.spinner("Analyzing over/under for all games..."):
    for idx, game in enumerate(games):
        # Check if predictions already exist in the game (from flattened precomputed data)
        if 'predicted_total' in game and game.get('predicted_total') is not None:
            # Use existing predictions
            enriched_game = game
            predicted_total = game['predicted_total']
            total_pred = {'prediction': predicted_total, 'confidence': 'Medium'}
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
            
            # Extract total prediction
            total_pred = predictions.get('total', {})
            if not total_pred or 'prediction' not in total_pred:
                continue
            
            predicted_total = total_pred['prediction']
        
        # Get actual total line if available (betting_over_under is the field from enriched data)
        actual_total = enriched_game.get('betting_over_under')
        if actual_total is None:
            # Try alternative field names as fallback
            actual_total = enriched_game.get('total') or enriched_game.get('over_under')
        if actual_total is None:
            processing_debug.append(f"No total for {enriched_game.get('away_team', '?')} @ {enriched_game.get('home_team', '?')}")
            continue  # Skip if no total line
        
        # Calculate edge
        edge = abs(predicted_total - actual_total)
        
        # Get pace information from stats (handle None case for flattened data)
        home_stats = enriched_game.get('home_stats') or {}
        away_stats = enriched_game.get('away_stats') or {}
        home_tempo = home_stats.get('pace', 70)
        away_tempo = away_stats.get('pace', 70)
        avg_tempo = (home_tempo + away_tempo) / 2
        tempo_diff = abs(home_tempo - away_tempo)
        
        # Apply tempo filter
        if tempo_filter == "High Tempo (72+ possessions)" and avg_tempo < 72:
            continue
        elif tempo_filter == "Medium Tempo (68-72)" and (avg_tempo < 68 or avg_tempo > 72):
            continue
        elif tempo_filter == "Low Tempo (<68)" and avg_tempo >= 68:
            continue
        elif tempo_filter == "Pace Mismatch (5+ difference)" and tempo_diff < 5:
            continue
        
        # Apply filters
        if ranked_only and not (game.get('home_rank') or game.get('away_rank')):
            continue
        
        if actual_total < total_range[0] or actual_total > total_range[1]:
            continue
        
        if edge < min_edge:
            continue
        
        # Determine pick
        if predicted_total > actual_total:
            pick = "OVER"
        else:
            pick = "UNDER"
        
        ou_picks.append({
            'matchup': f"{enriched_game['away_team']} @ {enriched_game['home_team']}",
            'away_team': enriched_game['away_team'],
            'home_team': enriched_game['home_team'],
            'line': actual_total,
            'projected': predicted_total,
            'edge': edge,
            'pick': pick,
            'avg_tempo': avg_tempo,
            'tempo_diff': tempo_diff,
            'away_rank': game.get('away_rank'),
            'home_rank': game.get('home_rank'),
            'confidence': total_pred.get('confidence', 'Medium')
        })

# Show debugging info
if processing_debug:
    with st.expander("⚠️ Debugging Info: Games Without Totals", expanded=False):
        st.write(f"{len(processing_debug)} games skipped due to missing total data:")
        st.info("💡 Betting lines may not be available yet for upcoming games. Run `python fetch_live_odds.py` to fetch current odds, then `python generate_predictions.py` to refresh predictions with betting lines.")
        for msg in processing_debug[:20]:  # Show first 20
            st.caption(msg)

if not ou_picks:
    st.warning("No games match the current filter criteria.")
    st.info(f"**Summary:** Processed {len(games)} games, {len(processing_debug)} had no total data")
    
    if len(processing_debug) > len(games) * 0.8:  # If more than 80% missing data
        st.error("🔴 Most games are missing betting lines. To fix this:")
        st.code("""# Fetch current odds and regenerate predictions
python fetch_live_odds.py
python generate_predictions.py
""")
    else:
        st.info("Try adjusting the filters (lower minimum edge, wider total range, disable ranked-only filter, change tempo filter)")
    st.stop()

# Convert to DataFrame
df = pd.DataFrame(ou_picks)

# Display summary metrics
st.header("📊 Over/Under Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Games", len(df))

with col2:
    avg_edge = df['edge'].mean()
    st.metric("Average Edge", f"{avg_edge:.1f} pts")

with col3:
    over_picks = len(df[df['pick'] == 'OVER'])
    st.metric("Over Picks", over_picks)

with col4:
    under_picks = len(df[df['pick'] == 'UNDER'])
    st.metric("Under Picks", under_picks)

# Side-by-side best overs and unders
st.header("🔥 Best Picks")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🔺 Best Overs")
    overs = df[df['pick'] == 'OVER'].nlargest(10, 'edge')
    
    if len(overs) == 0:
        st.info("No over picks found with current filters.")
    else:
        for idx, game in overs.iterrows():
            with st.container():
                st.markdown(f"**{game['matchup']}**")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Line", f"{game['line']:.1f}")
                
                with metric_col2:
                    st.metric("Projected", f"{game['projected']:.1f}")
                
                with metric_col3:
                    st.metric("Edge", f"+{game['edge']:.1f}")
                
                st.caption(f"🏃 Avg Tempo: {game['avg_tempo']:.1f} possessions")
                st.divider()

with col2:
    st.subheader("🔻 Best Unders")
    unders = df[df['pick'] == 'UNDER'].nlargest(10, 'edge')
    
    if len(unders) == 0:
        st.info("No under picks found with current filters.")
    else:
        for idx, game in unders.iterrows():
            with st.container():
                st.markdown(f"**{game['matchup']}**")
                
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                
                with metric_col1:
                    st.metric("Line", f"{game['line']:.1f}")
                
                with metric_col2:
                    st.metric("Projected", f"{game['projected']:.1f}")
                
                with metric_col3:
                    st.metric("Edge", f"+{game['edge']:.1f}")
                
                st.caption(f"🏃 Avg Tempo: {game['avg_tempo']:.1f} possessions")
                st.divider()

# Full picks table
st.header("📋 All Over/Under Picks")

# Format the display dataframe
display_df = df.copy()
display_df['Matchup'] = display_df['matchup']
display_df['Line'] = display_df['line'].apply(lambda x: f"{x:.1f}")
display_df['Projected'] = display_df['projected'].apply(lambda x: f"{x:.1f}")
display_df['Edge'] = display_df['edge'].apply(lambda x: f"{x:.1f}")
display_df['Pick'] = display_df['pick']
display_df['Avg Tempo'] = display_df['avg_tempo'].apply(lambda x: f"{x:.1f}")
display_df['Confidence'] = display_df['confidence']

st.dataframe(
    display_df[['Matchup', 'Line', 'Projected', 'Edge', 'Pick', 'Avg Tempo', 'Confidence']],
    use_container_width=True,
    hide_index=True
)

# Visualization: Projected Total vs Line
st.header("📈 Projected Total vs Current Line")

fig = px.scatter(
    df,
    x='line',
    y='projected',
    color='pick',
    size='edge',
    hover_data=['matchup', 'avg_tempo', 'confidence'],
    title="Over/Under Analysis: Projected Total vs Current Line",
    labels={
        'line': 'Current O/U Line',
        'projected': 'Projected Total',
        'pick': 'Pick',
        'edge': 'Edge'
    },
    color_discrete_map={'OVER': '#00CC96', 'UNDER': '#EF553B'}
)

# Add diagonal line (where projected = actual)
min_val = min(df['line'].min(), df['projected'].min())
max_val = max(df['line'].max(), df['projected'].max())
fig.add_shape(
    type="line",
    x0=min_val, y0=min_val,
    x1=max_val, y1=max_val,
    line=dict(color="gray", dash="dash", width=2),
    name="Line = Projection"
)

fig.update_layout(
    height=600,
    xaxis_title="Current Over/Under Line",
    yaxis_title="Projected Total Points",
    hovermode='closest'
)

st.plotly_chart(fig, use_container_width=True)

# Tempo analysis
st.header("🏃 Tempo Analysis")

fig2 = px.scatter(
    df,
    x='avg_tempo',
    y='projected',
    color='pick',
    size='edge',
    hover_data=['matchup', 'line'],
    title="Tempo vs Projected Total",
    labels={
        'avg_tempo': 'Average Tempo (possessions per game)',
        'projected': 'Projected Total Points',
        'pick': 'Pick',
        'edge': 'Edge'
    },
    color_discrete_map={'OVER': '#00CC96', 'UNDER': '#EF553B'}
)

fig2.update_layout(
    height=500,
    xaxis_title="Average Game Tempo",
    yaxis_title="Projected Total Points"
)

st.plotly_chart(fig2, use_container_width=True)

# Edge distribution
st.header("📊 Edge Distribution")

fig3 = go.Figure()

fig3.add_trace(go.Histogram(
    x=df[df['pick'] == 'OVER']['edge'],
    name='Over',
    marker_color='#00CC96',
    opacity=0.7
))

fig3.add_trace(go.Histogram(
    x=df[df['pick'] == 'UNDER']['edge'],
    name='Under',
    marker_color='#EF553B',
    opacity=0.7
))

fig3.update_layout(
    title="Distribution of Edges by Pick Type",
    xaxis_title="Edge (points)",
    yaxis_title="Number of Games",
    barmode='overlay',
    height=400
)

st.plotly_chart(fig3, use_container_width=True)

# Footer
st.markdown("---")
st.caption("💡 **Tip**: High-tempo games tend to have higher scoring, while low-tempo games favor unders. Pace mismatches can create unique opportunities.")
