"""
Tournament Bracket Visualization Page

Interactive full bracket display with Monte Carlo simulation results.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from bracket_simulation import (
    load_real_tournament_bracket,
    create_bracket_from_data,
    create_predictor_from_models,
    simulate_bracket,
    run_single_simulation
)
from data_tools.efficiency_loader import EfficiencyDataLoader
import plotly.io as pio
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Layout configuration: single place to tune visual constants
LAYOUT_CONFIG = {
    'y_spacing': 32,
    'x_round_spacing': 140,
    # right-most visible coordinate (matches fig.update_layout xaxis.range)
    'xaxis_range': [0, 1875],
    'yaxis_range': [0, 1050],
    'width': 2200,
    'height': 1400,
    'left_region_x': 50,
    'right_region_x': 1825,
    'region_layout': {
        'South': {'x_start': 50, 'y_start': 550, 'direction': 1, 'label_pos': 'top-left'},
        'East': {'x_start': 50, 'y_start': 30, 'direction': 1, 'label_pos': 'bottom-left'},
        'Midwest': {'x_start': 1825, 'y_start': 550, 'direction': -1, 'label_pos': 'top-right'},
        'West': {'x_start': 1825, 'y_start': 30, 'direction': -1, 'label_pos': 'bottom-right'}
    },
    # Final Four columns (left and right semicenters)
    'final_four_x': 650,
    'final_four_right_x': 1230,
    # championship center (None = computed as midpoint)
    'center_x': None,

    # Additional visual knobs
    'label_outset': 25,
    'region_label_vertical_offset': 50,
    'team_box_width': 100,
    'team_text_offset': 50,
    'team_marker_size': 28,
    'team_text_size': 12,
    'winner_label_x_offset': 80,
    'winner_label_y_offset': 8,
    'winner_text_size': 10,
    'final_four_label_size': 24,
    'finalist_name_size': 9,
    'champion_name_size': 14,
    'trophy_size': 60,
    'final_four_name_x_offset': 50,
    'finalist_name_x_offset': 50,
    'round_connector_offset': 10,
    # Marker sizes by round (r2=R32, r3=Sweet16, r4=Elite8, r5=FinalFour/region champ)
    'marker_sizes': {
        'r2': 16,
        'r3': 15,
        'r4': 14,
        'r5': 18,
        'final_four': 16,
        'finalist': 14,
        'champion': 18
    }
    ,
    'team_marker_inner_offset_left': 25,
    'team_marker_inner_offset_right': 75
}


st.set_page_config(
    page_title="Tournament Bracket",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 March Madness Tournament Bracket")
st.markdown("*Monte Carlo simulation of full tournament outcomes*")

# Sidebar controls
st.sidebar.header("Bracket Controls")

tournament_year = st.sidebar.selectbox(
    "Tournament Year",
    [2025, 2024, 2023],
    index=0
)

# Probability threshold filter
min_prob_filter = st.sidebar.slider(
    "Minimum Win Probability to Show",
    0.0, 1.0, 0.0, 0.05,
    help="Filter out low-probability predictions"
)

show_upsets_only = st.sidebar.checkbox(
    "Show Only Upset Predictions",
    help="Display only games where lower seed is favored"
)

# Visualization mode
viz_mode = st.sidebar.radio(
    "Visualization Mode",
    ["Visual Bracket", "Probability Heatmap", "Text Bracket", "All Views"],
    index=0,
    help="Choose how to display the bracket"
)

# Team search functionality
st.sidebar.divider()
st.sidebar.subheader("🔍 Team Search")

# This will be populated after sim_results are loaded
selected_team_placeholder = st.sidebar.empty()

# Round filter
selected_round = st.sidebar.selectbox(
    "Jump to round:",
    ['All Rounds', 'Round of 32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship'],
    help="Filter display by tournament round"
)


@st.cache_resource(ttl=3600)
def load_precomputed_bracket(year: int):
    """Load pre-computed bracket simulation results."""
    import json
    from pathlib import Path
    from collections import namedtuple
    
    try:
        # Try loading pre-computed results first
        precomputed_file = Path(f'data_files/precomputed_brackets/bracket_{year}.json')
        
        if precomputed_file.exists():
            with open(precomputed_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct simulation results with Team objects
            Team = namedtuple('Team', ['name', 'seed', 'region'])
            simulation_results = {}
            
            for team_id, stats in data['simulation_results'].items():
                team_data = stats['team']
                simulation_results[team_id] = {
                    'team': Team(
                        name=team_data['name'],
                        seed=team_data['seed'],
                        region=team_data['region']
                    ),
                    'round_32_prob': stats.get('round_32_prob', 0.0),
                    'sweet_16_prob': stats.get('sweet_16_prob', 0.0),
                    'elite_8_prob': stats.get('elite_8_prob', 0.0),
                    'final_four_prob': stats.get('final_four_prob', 0.0),
                    'championship_prob': stats.get('championship_prob', 0.0),
                    'winner_prob': stats.get('winner_prob', 0.0)
                }
            
            return data['bracket_data'], simulation_results, True, data['num_simulations']
        
        # Fallback to live simulation if no pre-computed data
        return load_and_simulate_bracket_live(year, 1000)
        
    except Exception as e:
        st.error(f"Error loading pre-computed bracket: {e}")
        # Try live simulation as fallback
        try:
            return load_and_simulate_bracket_live(year, 1000)
        except:
            return None, None, False, 0


def load_and_simulate_bracket_live(year: int, num_sims: int):
    """Run live Monte Carlo simulation (fallback when pre-computed data unavailable)."""
    try:
        # Load tournament bracket
        bracket_data = load_real_tournament_bracket(year)
        
        # Load efficiency data for predictions
        efficiency_loader = EfficiencyDataLoader()
        kenpom_df = efficiency_loader.load_kenpom()
        bart_df = efficiency_loader.load_barttorvik()
        
        # Create bracket state and simulator
        bracket_state, simulator = create_bracket_from_data(bracket_data)
        
        # Create game predictor
        game_predictor = create_predictor_from_models(efficiency_data=kenpom_df)
        simulator.game_predictor = game_predictor
        
        # Run simulations
        simulation_results = simulator.simulate_bracket(bracket_state, num_simulations=num_sims)
        
        return bracket_data, simulation_results, True, num_sims
        
    except Exception as e:
        st.error(f"Error loading bracket: {e}")
        return None, None, False, 0


def show_team_path(sim_results: dict, team_name: str):
    """Show a specific team's predicted tournament path."""
    
    # Find the team
    team_stats = None
    for tid, stats in sim_results.items():
        if stats['team'].name == team_name:
            team_stats = stats
            break
    
    if not team_stats:
        st.warning(f"Team '{team_name}' not found in bracket.")
        return
    
    team = team_stats['team']
    st.subheader(f"📍 {team.name}'s Predicted Path")
    
    # Display round-by-round probabilities
    rounds = [
        ('Round of 32', 'round_32_prob'),
        ('Sweet 16', 'sweet_16_prob'),
        ('Elite 8', 'elite_8_prob'),
        ('Final Four', 'final_four_prob'),
        ('Championship Game', 'championship_prob'),
        ('Win Tournament', 'winner_prob')
    ]
    
    for round_name, prob_key in rounds:
        prob = team_stats.get(prob_key, 0.0)
        
        if prob > 0.5:
            st.success(f"✅ **{round_name}**: {prob:.1%} chance to advance")
        elif prob > 0.25:
            st.info(f"⚠️ **{round_name}**: {prob:.1%} chance to advance")
        elif prob > 0.05:
            st.warning(f"⚡ **{round_name}**: {prob:.1%} chance to advance (underdog)")
        else:
            st.error(f"❌ **{round_name}**: {prob:.1%} chance to advance")
            break  # Very low probability, likely eliminated


def show_probability_table(sim_results: dict):
    """Show sortable table of all team probabilities."""
    
    st.subheader("📊 Full Probability Table")
    
    # Create DataFrame
    rows = []
    for team_id, stats in sim_results.items():
        team = stats['team']
        rows.append({
            'Team': team.name,
            'Seed': team.seed,
            'Region': team.region,
            'R32': f"{stats.get('round_32_prob', 0):.1%}",
            'S16': f"{stats.get('sweet_16_prob', 0):.1%}",
            'E8': f"{stats.get('elite_8_prob', 0):.1%}",
            'FF': f"{stats.get('final_four_prob', 0):.1%}",
            'Finals': f"{stats.get('championship_prob', 0):.1%}",
            'Champ': f"{stats.get('winner_prob', 0):.1%}"
        })
    
    df = pd.DataFrame(rows)
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        ['Champ', 'FF', 'E8', 'S16', 'Seed'],
        key='prob_table_sort'
    )
    
    # Display table
    st.dataframe(
        df.sort_values(sort_by, ascending=(sort_by == 'Seed')),
        hide_index=True,
        width='stretch'
    )


# Load data
with st.spinner(f"Loading {tournament_year} tournament bracket..."):
    bracket_data, sim_results, success, actual_num_sims = load_precomputed_bracket(tournament_year)

if not success or not sim_results:
    st.warning("⚠️ Could not load real tournament data. Showing sample bracket for demonstration.")
    st.info("This is using synthetic data for development purposes. Real tournament data will be available during March Madness.")
else:
    # Show info about data source
    if actual_num_sims >= 5000:
        st.success(f"✓ Loaded pre-computed results ({actual_num_sims:,} simulations)")
    else:
        st.info(f"Running live simulation ({actual_num_sims:,} simulations)")
    # Continue with whatever data we have

# Add team search now that sim_results are loaded
if sim_results:
    # Get all team names
    all_teams = sorted([stats['team'].name for stats in sim_results.values()])
    
    # Team search in sidebar
    selected_team = selected_team_placeholder.selectbox(
        "Find a team:",
        [""] + all_teams,
        key="team_search"
    )
    
    # Show team path if selected
    if selected_team:
        st.divider()
        show_team_path(sim_results, selected_team)
        st.divider()


def render_half_bracket(sim_results: dict, region1: str, region2: str):
    """Render half of the bracket (2 regions meeting in Elite 8)."""
    
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    
    # Get teams for each region
    region1_teams = {tid: stats for tid, stats in sim_results.items() 
                     if stats['team'].region == region1}
    region2_teams = {tid: stats for tid, stats in sim_results.items() 
                     if stats['team'].region == region2}
    
    # Region 1 - Left side
    with col1:
        st.subheader(f"📍 {region1}")
        render_region_teams(region1_teams, "All Teams")
    
    with col2:
        st.caption("Sweet 16")
        render_region_teams(region1_teams, "sweet_16_prob", 0.15)
    
    # Sweet 16 / Elite 8 in center
    with col3:
        st.caption("Elite 8 / Final Four")
        render_region_teams(region1_teams, "elite_8_prob", 0.30)
        st.divider()
        render_region_teams(region2_teams, "elite_8_prob", 0.30)
    
    # Region 2 - Right side
    with col4:
        st.caption("Sweet 16")
        render_region_teams(region2_teams, "sweet_16_prob", 0.15)
    
    with col5:
        st.subheader(f"📍 {region2}")
        render_region_teams(region2_teams, "All Teams")


def render_region_teams(teams: dict, prob_key: str = None, min_prob: float = 0.0):
    """Render teams for a specific region and probability threshold."""
    
    # Sort by seed
    sorted_teams = sorted(
        teams.items(),
        key=lambda x: x[1]['team'].seed
    )
    
    for team_id, stats in sorted_teams:
        team = stats['team']
        
        # Get relevant probability
        if prob_key and prob_key != "All Teams":
            prob = stats.get(prob_key, 0.0)
            if prob < min_prob:
                continue
        else:
            # For "All Teams", show Final Four probability
            prob = stats.get('final_four_prob', 0.0)
        
        # Color coding based on probability
        if prob > 0.8:
            color = "🟢"  # High confidence
        elif prob > 0.6:
            color = "🟡"  # Medium confidence
        elif prob > 0.4:
            color = "🟠"  # Moderate
        else:
            color = "🔴"  # Low confidence
        
        # Show upset indicator for high seeds with good chances
        upset_marker = "⚡" if team.seed >= 10 and prob > 0.3 else ""
        
        # Display team
        if prob_key and prob_key != "All Teams":
            st.markdown(f"{color} **{team.seed}** {team.name} {upset_marker} *({prob:.1%})*")
        else:
            st.markdown(f"**{team.seed}** {team.name}")


def create_probability_heatmap(sim_results: dict, top_n: int = 32) -> go.Figure:
    """Create heatmap showing advancement probabilities for all teams."""
    
    # Sort teams by championship probability
    sorted_teams = sorted(
        sim_results.items(),
        key=lambda x: x[1].get('winner_prob', 0),
        reverse=True
    )[:top_n]
    
    # Prepare data
    team_names = []
    prob_matrix = []
    
    for team_id, stats in sorted_teams:
        team = stats['team']
        team_names.append(f"({team.seed}) {team.name}")
        
        prob_matrix.append([
            stats.get('round_32_prob', 0),
            stats.get('sweet_16_prob', 0),
            stats.get('elite_8_prob', 0),
            stats.get('final_four_prob', 0),
            stats.get('championship_prob', 0),
            stats.get('winner_prob', 0)
        ])
    
    rounds = ['R32', 'Sweet 16', 'Elite 8', 'Final Four', 'Finals', 'Champion']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=rounds,
        y=team_names,
        colorscale='RdYlGn',
        text=[[f"{p:.0%}" for p in row] for row in prob_matrix],
        texttemplate="%{text}",
        textfont={"size": 9},
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Round: %{x}<br>"
            "Probability: %{z:.1%}<extra></extra>"
        ),
        colorbar=dict(
            title="Win %",
            tickformat=".0%"
        )
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Teams - Advancement Probabilities by Round",
        xaxis_title="Tournament Round",
        yaxis_title="Team (Seed) Name",
        height=max(600, top_n * 20),  # Scale height with number of teams
        font=dict(size=10)
    )
    
    return fig


def create_visual_bracket(sim_results: dict) -> go.Figure:
    """Create visual bracket layout that looks like traditional March Madness bracket."""
    
    fig = go.Figure()
    
    # Get teams by region and seed
    regions = {'East': [], 'West': [], 'South': [], 'Midwest': []}
    for team_id, stats in sim_results.items():
        team = stats['team']
        # Skip teams with TBD region (not yet determined)
        if team.region in regions:
            regions[team.region].append((team.seed, team.name, stats))
    
    # Sort by seed within each region
    for region in regions:
        regions[region].sort(key=lambda x: x[0])
    
    # Layout parameters for traditional bracket (pulled from LAYOUT_CONFIG)
    cfg = LAYOUT_CONFIG
    y_spacing = cfg.get('y_spacing', 32)
    x_round_spacing = cfg.get('x_round_spacing', 140)
    bracket_height = cfg.get('bracket_height', 16 * y_spacing)

    # Starting positions for each region (use the config block)
    region_layout = cfg.get('region_layout', {
        'South': {'x_start': cfg.get('left_region_x', 50), 'y_start': 550, 'direction': 1, 'label_pos': 'top-left'},
        'East': {'x_start': cfg.get('left_region_x', 50), 'y_start': 30, 'direction': 1, 'label_pos': 'bottom-left'},
        'Midwest': {'x_start': cfg.get('right_region_x', 1825), 'y_start': 550, 'direction': -1, 'label_pos': 'top-right'},
        'West': {'x_start': cfg.get('right_region_x', 1825), 'y_start': 30, 'direction': -1, 'label_pos': 'bottom-right'}
    })
    
    def get_color(prob: float) -> str:
        """Get color based on win probability."""
        if prob > 0.7:
            return '#27ae60'  # Green
        elif prob > 0.5:
            return '#f39c12'  # Orange
        elif prob > 0.3:
            return '#e67e22'  # Dark orange
        else:
            return '#c0392b'  # Red
    
    def draw_matchup_bracket(x1, y1, y2, x2, y_mid, color='#95a5a6'):
        """Draw the bracket lines connecting two teams to next round."""
        # Horizontal line from team 1
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y1, y1],
            mode='lines', line=dict(width=1.5, color=color),
            showlegend=False, hoverinfo='skip'
        ))
        # Horizontal line from team 2
        fig.add_trace(go.Scatter(
            x=[x1, x2], y=[y2, y2],
            mode='lines', line=dict(width=1.5, color=color),
            showlegend=False, hoverinfo='skip'
        ))
        # Vertical connecting line
        fig.add_trace(go.Scatter(
            x=[x2, x2], y=[y1, y2],
            mode='lines', line=dict(width=1.5, color=color),
            showlegend=False, hoverinfo='skip'
        ))
        # Horizontal to next position at midpoint
        fig.add_trace(go.Scatter(
            x=[x2, x2], y=[y_mid, y_mid],
            mode='lines', line=dict(width=1.5, color=color),
            showlegend=False, hoverinfo='skip'
        ))
    
    # Store region exit positions for Final Four connections
    region_exits = {}
    
    # Draw each region
    for region_name, teams in regions.items():
        layout = region_layout[region_name]
        x_start = layout['x_start']
        y_start = layout['y_start']
        direction = layout['direction']
        
        # Add region label (use configurable offsets)
        label_x = x_start - cfg.get('label_outset', 25) if direction == 1 else x_start + cfg.get('label_outset', 25)
        label_angle = -90 if direction == 1 else 90
        fig.add_annotation(
            x=label_x, y=y_start + bracket_height/2 + cfg.get('region_label_vertical_offset', 50),
            text=f"<b>{region_name.upper()}</b>",
            showarrow=False,
            font=dict(size=20, color='#2c3e50', family='Arial Black'),
            textangle=label_angle
        )
        
        # Round 1: First Four (all 16 teams)
        round1_positions = []
        for i, (seed, name, stats) in enumerate(teams):
            y_pos = y_start + (i * y_spacing)
            round1_positions.append(y_pos)
            
            # Team box (box width comes from config)
            box_x = x_start if direction == 1 else x_start - cfg.get('team_box_width', 100)
            
            # Draw seed box
            fig.add_trace(go.Scatter(
                x=[box_x + (cfg.get('team_marker_inner_offset_left', 25) if direction == 1 else cfg.get('team_marker_inner_offset_right', 75))],
                y=[y_pos],
                mode='markers+text',
                marker=dict(size=cfg.get('team_marker_size', 28), color='white', line=dict(width=1.5, color='#bdc3c7')),
                text=f"{seed}",
                textfont=dict(size=cfg.get('team_text_size', 12), color='#34495e', family='Arial'),
                textposition='middle center',
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>Seed: {seed}<br>R32 Prob: {stats.get('round_32_prob', 1.0):.1%}<extra></extra>"
            ))
            
            # Team name
            text_anchor = 'left' if direction == 1 else 'right'
            text_x = box_x + cfg.get('team_text_offset', 50)
            fig.add_annotation(
                x=text_x, y=y_pos,
                text=name[:25],
                showarrow=False,
                font=dict(size=cfg.get('team_text_size', 12), color='#2c3e50'),
                xanchor=text_anchor,
                yanchor='middle'
            )
        
        # Round 2: Round of 32 (8 games)
        round2_x = x_start + (x_round_spacing * direction)
        round2_positions = []
        round2_winners = []  # Track winners for next round
        for i in range(8):
            y1 = round1_positions[i*2]
            y2 = round1_positions[i*2 + 1]
            y_mid = (y1 + y2) / 2
            round2_positions.append(y_mid)
            
            # Draw bracket connecting lines (connector offset configurable)
            bracket_x = round2_x - cfg.get('round_connector_offset', 10) if direction == 1 else round2_x + cfg.get('round_connector_offset', 10)
            draw_matchup_bracket(bracket_x, y1, y2, round2_x, y_mid)
            
            # Winner marker (team with higher Sweet 16 prob)
            teams_in_matchup = [teams[i*2], teams[i*2 + 1]]
            winner = max(teams_in_matchup, key=lambda t: t[2].get('sweet_16_prob', 0))
            round2_winners.append(winner)  # Store winner
            prob = winner[2].get('sweet_16_prob', 0)
            
            marker_x = round2_x
            fig.add_trace(go.Scatter(
                x=[marker_x],
                y=[y_mid],
                mode='markers',
                marker=dict(size=cfg.get('marker_sizes', {}).get('r2', 16), color=get_color(prob), line=dict(width=1.5, color='white')),
                showlegend=False,
                hovertemplate=f"<b>{winner[1]}</b><br>S16: {prob:.1%}<extra></extra>"
            ))
            
            # Add winner name above the line
            text_anchor = 'left' if direction == 1 else 'right'
            label_x = round2_x + (cfg.get('winner_label_x_offset', 80) if direction == 1 else -cfg.get('winner_label_x_offset', 80))
            fig.add_annotation(
                x=label_x, y=y_mid + cfg.get('winner_label_y_offset', 8),
                text=winner[1][:12],
                showarrow=False,
                font=dict(size=cfg.get('winner_text_size', 10), color='#2c3e50'),
                xanchor=text_anchor,
                yanchor='bottom'
            )
        
        # Round 3: Sweet 16 (4 games)
        round3_x = x_start + (2 * x_round_spacing * direction)
        round3_positions = []
        round3_winners = []  # Track winners for next round
        for i in range(4):
            y1 = round2_positions[i*2]
            y2 = round2_positions[i*2 + 1]
            y_mid = (y1 + y2) / 2
            round3_positions.append(y_mid)
            
            draw_matchup_bracket(round2_x, y1, y2, round3_x, y_mid)
            
            # Winner from actual Round 2 matchup
            teams_in_matchup = [round2_winners[i*2], round2_winners[i*2 + 1]]
            winner = max(teams_in_matchup, key=lambda t: t[2].get('elite_8_prob', 0))
            round3_winners.append(winner)  # Store winner
            prob = winner[2].get('elite_8_prob', 0)
            
            fig.add_trace(go.Scatter(
                x=[round3_x],
                y=[y_mid],
                mode='markers',
                marker=dict(size=cfg.get('marker_sizes', {}).get('r3', 15), color=get_color(prob), line=dict(width=1.5, color='white')),
                showlegend=False,
                hovertemplate=f"<b>{winner[1]}</b><br>E8: {prob:.1%}<extra></extra>"
            ))
            
            # Add winner name above the line
            text_anchor = 'left' if direction == 1 else 'right'
            label_x = round3_x + (cfg.get('winner_label_x_offset', 80) if direction == 1 else -cfg.get('winner_label_x_offset', 80))
            fig.add_annotation(
                x=label_x, y=y_mid + cfg.get('winner_label_y_offset', 8),
                text=winner[1][:12],
                showarrow=False,
                font=dict(size=cfg.get('winner_text_size', 10), color='#2c3e50'),
                xanchor=text_anchor,
                yanchor='bottom'
            )
        
        # Round 4: Elite 8 (2 games)
        round4_x = x_start + (3 * x_round_spacing * direction)
        round4_positions = []
        round4_winners = []  # Track winners for next round
        for i in range(2):
            y1 = round3_positions[i*2]
            y2 = round3_positions[i*2 + 1]
            y_mid = (y1 + y2) / 2
            round4_positions.append(y_mid)
            
            draw_matchup_bracket(round3_x, y1, y2, round4_x, y_mid)
            
            # Winner from actual Round 3 matchup
            teams_in_matchup = [round3_winners[i*2], round3_winners[i*2 + 1]]
            winner = max(teams_in_matchup, key=lambda t: t[2].get('final_four_prob', 0))
            round4_winners.append(winner)  # Store winner
            prob = winner[2].get('final_four_prob', 0)
            
            fig.add_trace(go.Scatter(
                x=[round4_x],
                y=[y_mid],
                mode='markers',
                marker=dict(size=cfg.get('marker_sizes', {}).get('r4', 14), color=get_color(prob), line=dict(width=1.5, color='white')),
                showlegend=False,
                hovertemplate=f"<b>{winner[1]}</b><br>FF: {prob:.1%}<extra></extra>"
            ))
            
            # Add winner name above the line
            text_anchor = 'left' if direction == 1 else 'right'
            label_x = round4_x + (cfg.get('winner_label_x_offset', 80) if direction == 1 else -cfg.get('winner_label_x_offset', 80))
            fig.add_annotation(
                x=label_x, y=y_mid + cfg.get('winner_label_y_offset', 8),
                text=winner[1][:12],
                showarrow=False,
                font=dict(size=cfg.get('winner_text_size', 10), color='#2c3e50'),
                xanchor=text_anchor,
                yanchor='bottom'
            )
        
        # Round 5: Final Four
        round5_x = x_start + (4 * x_round_spacing * direction)
        y1 = round4_positions[0]
        y2 = round4_positions[1]
        y_final = (y1 + y2) / 2
        
        draw_matchup_bracket(round4_x, y1, y2, round5_x, y_final)
        
        # Region champion (winner from actual Round 4 matchup)
        teams_in_matchup = [round4_winners[0], round4_winners[1]]
        region_champ = max(teams_in_matchup, key=lambda t: t[2].get('final_four_prob', 0))
        prob = region_champ[2].get('final_four_prob', 0)
        
        fig.add_trace(go.Scatter(
            x=[round5_x],
            y=[y_final],
            mode='markers',
            marker=dict(size=cfg.get('marker_sizes', {}).get('r5', 18), color=get_color(prob), line=dict(width=2, color='white')),
            showlegend=False,
            hovertemplate=f"<b>{region_champ[1]}</b><br>{region_name}<br>FF: {prob:.1%}<extra></extra>"
        ))
        
        # Store this region's exit position
        region_exits[region_name] = {'x': round5_x, 'y': y_final}
    
    # Final Four meeting in center
    # `center_x` will be computed after the Final Four X positions are set
    center_x = None
    
    # Get Final Four teams (one from each region)
    south_ff = sorted(regions['South'], key=lambda t: t[2].get('final_four_prob', 0), reverse=True)[0]
    east_ff = sorted(regions['East'], key=lambda t: t[2].get('final_four_prob', 0), reverse=True)[0]
    midwest_ff = sorted(regions['Midwest'], key=lambda t: t[2].get('final_four_prob', 0), reverse=True)[0]
    west_ff = sorted(regions['West'], key=lambda t: t[2].get('final_four_prob', 0), reverse=True)[0]
    
    # South vs East (left semifinal)
    # Use actual exit positions from regions to calculate midpoint, but use consistent spacing
    south_exit = region_exits['South']
    east_exit = region_exits['East']
    left_semi_y = (south_exit['y'] + east_exit['y']) / 2
    # Use consistent spacing like other rounds (2x y_spacing for semifinal matchup)
    south_y = left_semi_y + (y_spacing * 2)
    east_y = left_semi_y - (y_spacing * 2)
    
    # Draw lines from regions to Final Four (positions configurable)
    final_four_x = cfg.get('final_four_x', 650)
    # Connect from actual region exit positions to Final Four team positions
    fig.add_trace(go.Scatter(
        x=[south_exit['x'], final_four_x], y=[south_exit['y'], south_exit['y']],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[final_four_x, final_four_x], y=[south_exit['y'], south_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[east_exit['x'], final_four_x], y=[east_exit['y'], east_exit['y']],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[final_four_x, final_four_x], y=[east_exit['y'], east_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    # Draw lines from Final Four teams to winner marker
    fig.add_trace(go.Scatter(
        x=[final_four_x, final_four_x], y=[south_y, left_semi_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[final_four_x, final_four_x], y=[east_y, left_semi_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Show South team in Final Four
    fig.add_trace(go.Scatter(
        x=[final_four_x],
        y=[south_y],
        mode='markers+text',
        marker=dict(size=cfg.get('marker_sizes', {}).get('final_four', 16), color=get_color(south_ff[2].get('final_four_prob', 0)), line=dict(width=2, color='white')),
        text=f"{south_ff[0]}",
        textfont=dict(size=cfg.get('team_text_size', 12), color='white'),
        showlegend=False,
        hovertemplate=f"<b>{south_ff[1]}</b><br>South<br>FF: {south_ff[2].get('final_four_prob', 0):.1%}<extra></extra>"
    ))
    fig.add_annotation(
        x=final_four_x - cfg.get('final_four_name_x_offset', 50), y=south_y,
        text=south_ff[1][:15],
        showarrow=False,
        font=dict(size=cfg.get('team_text_size', 12), color='#2c3e50'),
        xanchor='right'
    )
    
    # Show East team in Final Four
    fig.add_trace(go.Scatter(
        x=[final_four_x],
        y=[east_y],
        mode='markers+text',
        marker=dict(size=cfg.get('marker_sizes', {}).get('final_four', 16), color=get_color(east_ff[2].get('final_four_prob', 0)), line=dict(width=2, color='white')),
        text=f"{east_ff[0]}",
        textfont=dict(size=cfg.get('team_text_size', 12), color='white'),
        showlegend=False,
        hovertemplate=f"<b>{east_ff[1]}</b><br>East<br>FF: {east_ff[2].get('final_four_prob', 0):.1%}<extra></extra>"
    ))
    fig.add_annotation(
        x=final_four_x - cfg.get('final_four_name_x_offset', 50), y=east_y,
        text=east_ff[1][:15],
        showarrow=False,
        font=dict(size=cfg.get('team_text_size', 12), color='#2c3e50'),
        xanchor='right'
    )
    
    # Left semifinal winner
    left_winner = south_ff if south_ff[2].get('championship_prob', 0) > east_ff[2].get('championship_prob', 0) else east_ff
    fig.add_trace(go.Scatter(
        x=[final_four_x],
        y=[left_semi_y],
        mode='markers',
        marker=dict(size=cfg.get('marker_sizes', {}).get('finalist', 14), color=get_color(left_winner[2].get('championship_prob', 0)), line=dict(width=1.5, color='white')),
        showlegend=False,
        hovertemplate=f"<b>{left_winner[1]}</b><br>Finals: {left_winner[2].get('championship_prob', 0):.1%}<extra></extra>"
    ))
    
    # Midwest vs West (right semifinal)
    # Use actual exit positions from regions to calculate midpoint, but use consistent spacing
    midwest_exit = region_exits['Midwest']
    west_exit = region_exits['West']
    right_semi_y = (midwest_exit['y'] + west_exit['y']) / 2
    # Use consistent spacing like other rounds (2x y_spacing for semifinal matchup)
    midwest_y = right_semi_y + (y_spacing * 2)
    west_y = right_semi_y - (y_spacing * 2)
    
    # Connect from actual region exit positions to Final Four team positions
    final_four_right_x = cfg.get('final_four_right_x', 1250)

    # Compute championship center between the two Final Four columns
    # Championship center: explicit config overrides automatic midpoint
    cfg_center = cfg.get('center_x', None)
    if cfg_center is not None:
        center_x = cfg_center
    elif final_four_x is not None and final_four_right_x is not None:
        center_x = (final_four_x + final_four_right_x) / 2
    else:
        center_x = cfg.get('default_center_x', 900)
    fig.add_trace(go.Scatter(
        x=[midwest_exit['x'], final_four_right_x], y=[midwest_exit['y'], midwest_exit['y']],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[final_four_right_x, final_four_right_x], y=[midwest_exit['y'], midwest_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[west_exit['x'], final_four_right_x], y=[west_exit['y'], west_exit['y']],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[final_four_right_x, final_four_right_x], y=[west_exit['y'], west_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    # Draw lines from Final Four teams to winner marker
    fig.add_trace(go.Scatter(
        x=[final_four_right_x, final_four_right_x], y=[midwest_y, right_semi_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[final_four_right_x, final_four_right_x], y=[west_y, right_semi_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Show Midwest team in Final Four
    fig.add_trace(go.Scatter(
        x=[final_four_right_x],
        y=[midwest_y],
        mode='markers+text',
        marker=dict(size=cfg.get('marker_sizes', {}).get('final_four', 16), color=get_color(midwest_ff[2].get('final_four_prob', 0)), line=dict(width=2, color='white')),
        text=f"{midwest_ff[0]}",
        textfont=dict(size=cfg.get('team_text_size', 12), color='white'),
        showlegend=False,
        hovertemplate=f"<b>{midwest_ff[1]}</b><br>Midwest<br>FF: {midwest_ff[2].get('final_four_prob', 0):.1%}<extra></extra>"
    ))
    fig.add_annotation(
        x=final_four_right_x + cfg.get('final_four_name_x_offset', 50), y=midwest_y,
        text=midwest_ff[1][:15],
        showarrow=False,
        font=dict(size=cfg.get('team_text_size', 12), color='#2c3e50'),
        xanchor='left'
    )
    
    # Show West team in Final Four
    fig.add_trace(go.Scatter(
        x=[final_four_right_x],
        y=[west_y],
        mode='markers+text',
        marker=dict(size=cfg.get('marker_sizes', {}).get('final_four', 16), color=get_color(west_ff[2].get('final_four_prob', 0)), line=dict(width=2, color='white')),
        text=f"{west_ff[0]}",
        textfont=dict(size=cfg.get('team_text_size', 12), color='white'),
        showlegend=False,
        hovertemplate=f"<b>{west_ff[1]}</b><br>West<br>FF: {west_ff[2].get('final_four_prob', 0):.1%}<extra></extra>"
    ))
    fig.add_annotation(
        x=final_four_right_x + cfg.get('final_four_name_x_offset', 50), y=west_y,
        text=west_ff[1][:15],
        showarrow=False,
        font=dict(size=cfg.get('team_text_size', 12), color='#2c3e50'),
        xanchor='left'
    )
    
    # Right semifinal winner
    right_winner = midwest_ff if midwest_ff[2].get('championship_prob', 0) > west_ff[2].get('championship_prob', 0) else west_ff
    fig.add_trace(go.Scatter(
        x=[final_four_right_x],
        y=[right_semi_y],
        mode='markers',
        marker=dict(size=cfg.get('marker_sizes', {}).get('finalist', 14), color=get_color(right_winner[2].get('championship_prob', 0)), line=dict(width=1.5, color='white')),
        showlegend=False,
        hovertemplate=f"<b>{right_winner[1]}</b><br>Finals: {right_winner[2].get('championship_prob', 0):.1%}<extra></extra>"
    ))
    
    # Championship
    champ_y = (left_semi_y + right_semi_y) / 2
    
    # Draw bracket lines from both semifinals to championship
    # Left side: from final_four_x to center_x
    fig.add_trace(go.Scatter(
        x=[final_four_x, center_x-5], y=[left_semi_y, left_semi_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[center_x-5, center_x-5], y=[left_semi_y, champ_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[center_x-5, center_x], y=[champ_y, champ_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Right side: from final_four_right_x to center_x
    fig.add_trace(go.Scatter(
        x=[final_four_right_x, center_x+5], y=[right_semi_y, right_semi_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[center_x+5, center_x+5], y=[right_semi_y, champ_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=[center_x+5, center_x], y=[champ_y, champ_y],
        mode='lines', line=dict(width=1.5, color='#95a5a6'),
        showlegend=False, hoverinfo='skip'
    ))
    
    # Add Final Four label
    fig.add_annotation(
        x=center_x, y=champ_y + 100,
        text="<b>FINAL FOUR</b>",
        showarrow=False,
        font=dict(size=cfg.get('final_four_label_size', 24), color='#2c3e50', family='Arial Black')
    )
    
    # Championship game finalists
    fig.add_trace(go.Scatter(
        x=[center_x - cfg.get('finalist_name_x_offset', 50)],
        y=[champ_y],
        mode='markers',
        marker=dict(size=cfg.get('marker_sizes', {}).get('finalist', 14), color=get_color(left_winner[2].get('winner_prob', 0)), line=dict(width=2, color='white')),
        showlegend=False,
        hovertemplate=f"<b>{left_winner[1]}</b><br>Win: {left_winner[2].get('winner_prob', 0):.1%}<extra></extra>"
    ))
    fig.add_annotation(
        x=center_x - cfg.get('finalist_name_x_offset', 50), y=champ_y + 15,
        text=left_winner[1][:12],
        showarrow=False,
        font=dict(size=cfg.get('finalist_name_size', 9), color='#2c3e50'),
        xanchor='center',
        yanchor='bottom'
    )
    
    fig.add_trace(go.Scatter(
        x=[center_x + cfg.get('finalist_name_x_offset', 50)],
        y=[champ_y],
        mode='markers',
        marker=dict(size=cfg.get('marker_sizes', {}).get('finalist', 14), color=get_color(right_winner[2].get('winner_prob', 0)), line=dict(width=2, color='white')),
        showlegend=False,
        hovertemplate=f"<b>{right_winner[1]}</b><br>Win: {right_winner[2].get('winner_prob', 0):.1%}<extra></extra>"
    ))
    fig.add_annotation(
        x=center_x + cfg.get('finalist_name_x_offset', 50), y=champ_y + 15,
        text=right_winner[1][:12],
        showarrow=False,
        font=dict(size=cfg.get('finalist_name_size', 9), color='#2c3e50'),
        xanchor='center',
        yanchor='bottom'
    )
    
    # Champion
    champion = left_winner if left_winner[2].get('winner_prob', 0) > right_winner[2].get('winner_prob', 0) else right_winner
    fig.add_trace(go.Scatter(
        x=[center_x],
        y=[champ_y],
        mode='markers',
        marker=dict(size=cfg.get('marker_sizes', {}).get('champion', 18), color=get_color(champion[2].get('winner_prob', 0)), line=dict(width=2, color='gold')),
        showlegend=False,
        hovertemplate=f"<b>CHAMPION</b><br>{champion[1]}<br>{champion[2].get('winner_prob', 0):.1%}<extra></extra>"
    ))
    
    # Championship trophy
    fig.add_annotation(
        x=center_x, y=champ_y - 80,
        text="🏆",
        showarrow=False,
        font=dict(size=60)
    )
    
    # Champion name
    fig.add_annotation(
        x=center_x, y=champ_y - 140,
        text=f"<b>({champion[0]}) {champion[1]}</b><br>{champion[2].get('winner_prob', 0):.1%}",
        showarrow=False,
        font=dict(size=14, color='#2c3e50')
    )
    
    fig.update_layout(
        title={
            'text': "March Madness Tournament Bracket",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 28, 'color': '#2c3e50', 'family': 'Arial Black'}
        },
        showlegend=False,
        xaxis=dict(visible=False, range=cfg.get('xaxis_range', [0, 1875])),
        yaxis=dict(visible=False, range=cfg.get('yaxis_range', [0, 1050])),
        height=cfg.get('height', 1400),
        width=cfg.get('width', 2200),
        plot_bgcolor='#f8f9fa',
        margin=dict(l=40, r=40, t=80, b=40),
        hovermode='closest'
    )
    
    return fig


def render_final_four(sim_results: dict):
    """Render Final Four and Championship probabilities."""
    
    st.header("🏆 Final Four & Championship")
    
    # Get top Final Four candidates
    ff_teams = sorted(
        sim_results.items(),
        key=lambda x: x[1].get('final_four_prob', 0),
        reverse=True
    )[:8]  # Top 8 most likely
    
    # Get championship favorites
    champ_teams = sorted(
        sim_results.items(),
        key=lambda x: x[1].get('winner_prob', 0),
        reverse=True
    )[:5]  # Top 5
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.subheader("Most Likely Final Four")
        for team_id, stats in ff_teams:
            team = stats['team']
            prob = stats.get('final_four_prob', 0)
            st.markdown(f"**({team.seed}) {team.name}**")
            st.progress(prob, text=f"{prob:.1%}")
    
    with col2:
        st.subheader("🏆 Championship Favorites")
        
        # Show top team prominently
        if champ_teams:
            top_team_id, top_stats = champ_teams[0]
            top_team = top_stats['team']
            top_prob = top_stats.get('winner_prob', 0)
            
            st.success(f"### **({top_team.seed}) {top_team.name}**")
            st.metric("Win Probability", f"{top_prob:.1%}")
            st.metric("Final Four Probability", f"{top_stats.get('final_four_prob', 0):.1%}")
            
            st.divider()
            
            # Show rest
            for team_id, stats in champ_teams[1:]:
                team = stats['team']
                prob = stats.get('winner_prob', 0)
                st.markdown(f"**{team.seed}. {team.name}** - {prob:.1%}")
    
    with col3:
        st.subheader("Cinderella Watch")
        st.caption("Lower seeds with good chances")
        
        # Find high seeds with decent probabilities
        cinderellas = [
            (tid, stats) for tid, stats in sim_results.items()
            if stats['team'].seed is not None and stats['team'].seed >= 8 and stats.get('final_four_prob', 0) > 0.10
        ]
        cinderellas.sort(key=lambda x: x[1].get('final_four_prob', 0), reverse=True)
        
        for team_id, stats in cinderellas[:5]:
            team = stats['team']
            ff_prob = stats.get('final_four_prob', 0)
            win_prob = stats.get('winner_prob', 0)
            st.markdown(f"**#{team.seed} {team.name}**")
            st.caption(f"FF: {ff_prob:.1%} | Win: {win_prob:.1%}")


# Main bracket display
if sim_results:
    
    # Show visual bracket
    if viz_mode in ["Visual Bracket", "All Views"]:
        # st.header("🏀 Visual Tournament Bracket")
        
        bracket_fig = create_visual_bracket(sim_results)
        st.plotly_chart(bracket_fig, width='stretch')

        # Export / download controls (generate PNG and SVG via Plotly/Kaleido)
        try:
            png_bytes = pio.to_image(bracket_fig, format='png', width=LAYOUT_CONFIG.get('width', 2200), height=LAYOUT_CONFIG.get('height', 1400), scale=2)
        except Exception:
            png_bytes = None
        try:
            svg_bytes = pio.to_image(bracket_fig, format='svg', width=LAYOUT_CONFIG.get('width', 2200), height=LAYOUT_CONFIG.get('height', 1400), scale=1)
        except Exception:
            svg_bytes = None
        try:
            pdf_bytes = pio.to_image(bracket_fig, format='pdf', width=LAYOUT_CONFIG.get('width', 2200), height=LAYOUT_CONFIG.get('height', 1400), scale=1)
        except Exception:
            pdf_bytes = None

        col_export, col_caption = st.columns([1, 4])
        with col_export:
            if png_bytes:
                st.download_button("Download PNG", data=png_bytes, file_name="tournament_bracket.png", mime="image/png")
            else:
                st.info("PNG export unavailable (kaleido not installed)")

            if svg_bytes:
                st.download_button("Download SVG", data=svg_bytes, file_name="tournament_bracket.svg", mime="image/svg+xml")
            if pdf_bytes:
                st.download_button("Download PDF", data=pdf_bytes, file_name="tournament_bracket.pdf", mime="application/pdf")
            else:
                if not png_bytes and not svg_bytes:
                    # If none available, remind about kaleido
                    st.info("Image/PDF export unavailable. Install 'kaleido' to enable exports.")

        with col_caption:
            st.caption("💡 **How to read**: Each circle represents a team (number = seed). Color indicates advancement probability: Green = high, Orange = medium, Red = low. Hover over teams for details.")

        st.divider()
    
    # Show heatmap visualization
    if viz_mode in ["Probability Heatmap", "All Views"]:
        st.header("📊 Probability Statistics")
        
        # Number of teams to show
        top_n = st.slider(
            "Number of teams to display",
            min_value=16,
            max_value=64,
            value=32,
            step=8,
            key="heatmap_teams"
        )
        
        heatmap_fig = create_probability_heatmap(sim_results, top_n)
        st.plotly_chart(heatmap_fig, width='stretch')
        
        st.caption("💡 **How to read**: Each row is a team, each column is a tournament round. Green = high probability, Red = low probability.")
        
        st.divider()
    
    # Show text bracket
    if viz_mode in ["Text Bracket", "All Views"]:
        st.header("📝 Text Bracket View")
        
        # Top half: East and West
        st.subheader("East vs West")
        render_half_bracket(sim_results, 'East', 'West')
        
        st.divider()
        
        # Bottom half: South and Midwest
        st.subheader("South vs Midwest")
        render_half_bracket(sim_results, 'South', 'Midwest')
        
        st.divider()
    
    # Final Four and Championship (always show)
    render_final_four(sim_results)
    
    st.divider()
    
    # Detailed probability table
    with st.expander("📊 View Detailed Probability Table"):
        show_probability_table(sim_results)

else:
    st.error("Unable to load bracket data. Please check the configuration.")

# Footer with stats
st.divider()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Simulations Run", f"{actual_num_sims:,}")

with col2:
    if sim_results:
        total_teams = len(sim_results)
        st.metric("Teams in Bracket", total_teams)

with col3:
    if sim_results:
        # Calculate "chalk" probability (all 1 seeds in FF)
        one_seeds = [stats for stats in sim_results.values() if stats['team'].seed == 1]
        chalk_prob = 1.0
        for stats in one_seeds:
            chalk_prob *= stats.get('final_four_prob', 0)
        st.metric("All #1 Seeds in FF", f"{chalk_prob:.2%}")
