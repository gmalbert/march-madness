"""
Tournament Bracket Visualization Page

Interactive full bracket display with Monte Carlo simulation results.
"""

import math
import streamlit as st
import numpy as np
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
from upset_prediction import (
    generate_upset_watch_list,
    UpsetPredictor,
    create_training_data_from_csv,
    create_historical_training_data,
    HISTORICAL_UPSET_RATES,
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
        # x_start: leftmost column x for left regions, rightmost for right regions
        # y_start: seed-1 anchor (top of region for East/West, bottom for South/Midwest)
        # y_direction: -1 = bracket progresses downward (East/West, top half)
        #               1 = bracket progresses upward   (South/Midwest, bottom half)
        'East':    {'x_start': 50,   'y_start': 1020, 'y_direction': -1, 'direction': 1,  'label_pos': 'top-left'},
        'South':   {'x_start': 50,   'y_start': 510,  'y_direction': -1, 'direction': 1,  'label_pos': 'bottom-left'},
        'West':    {'x_start': 1825, 'y_start': 1020, 'y_direction': -1, 'direction': -1, 'label_pos': 'top-right'},
        'Midwest': {'x_start': 1825, 'y_start': 510,  'y_direction': -1, 'direction': -1, 'label_pos': 'bottom-right'},
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


st.title("🏀 March Madness Tournament Bracket")
st.markdown("*Monte Carlo simulation of full tournament outcomes*")

# Sidebar controls
st.sidebar.header("Bracket Controls")

tournament_year = st.sidebar.selectbox(
    "Tournament Year",
    [2026, 2025, 2024, 2023],
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
    ["Visual Bracket", "Interactive Grid", "Matchup Analysis", "Probability Heatmap", "Text Bracket", "All Views"],
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
        except Exception:
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


# ---------------------------------------------------------------------------
# Module-level short-name helper (used by bracket visual AND matchup analysis)
# ---------------------------------------------------------------------------
_SHORT_NAME_MAP = {
    'Duke Blue Devils': 'Duke',
    'UConn Huskies': 'UConn',
    'Michigan State Spartans': 'Michigan St.',
    'Kansas Jayhawks': 'Kansas',
    "St. John's Red Storm": "St. John's",
    'Louisville Cardinals': 'Louisville',
    'UCLA Bruins': 'UCLA',
    'Ohio State Buckeyes': 'Ohio State',
    'TCU Horned Frogs': 'TCU',
    'UCF Knights': 'UCF',
    'South Florida Bulls': 'South Florida',
    'Northern Iowa Panthers': 'Northern Iowa',
    'California Baptist Lancers': 'Cal Baptist',
    'North Dakota State Bison': 'NDSU',
    'Furman Paladins': 'Furman',
    'Siena Saints': 'Siena',
    'Arizona Wildcats': 'Arizona',
    'Purdue Boilermakers': 'Purdue',
    'Gonzaga Bulldogs': 'Gonzaga',
    'Arkansas Razorbacks': 'Arkansas',
    'Wisconsin Badgers': 'Wisconsin',
    'BYU Cougars': 'BYU',
    'Miami Hurricanes': 'Miami',
    'Villanova Wildcats': 'Villanova',
    'Utah State Aggies': 'Utah State',
    'Missouri Tigers': 'Missouri',
    'Texas Longhorns': 'Texas',
    'High Point Panthers': 'High Point',
    "Hawai'i Rainbow Warriors": "Hawai'i",
    'Kennesaw State Owls': 'Kennesaw St.',
    'Queens University Royals': 'Queens',
    'Long Island University Sharks': 'LIU',
    'Michigan Wolverines': 'Michigan',
    'Iowa State Cyclones': 'Iowa State',
    'Virginia Cavaliers': 'Virginia',
    'Alabama Crimson Tide': 'Alabama',
    'Texas Tech Red Raiders': 'Texas Tech',
    'Tennessee Volunteers': 'Tennessee',
    'Kentucky Wildcats': 'Kentucky',
    'Georgia Bulldogs': 'Georgia',
    'Saint Louis Billikens': 'Saint Louis',
    'Santa Clara Broncos': 'Santa Clara',
    'SMU Mustangs': 'SMU',
    'Akron Zips': 'Akron',
    'Hofstra Pride': 'Hofstra',
    'Wright State Raiders': 'Wright State',
    'Tennessee State Tigers': 'Tennessee St.',
    'Howard Bison': 'Howard',
    'UMBC Retrievers': 'UMBC',
    'Florida Gators': 'Florida',
    'Houston Cougars': 'Houston',
    'Illinois Fighting Illini': 'Illinois',
    'Nebraska Cornhuskers': 'Nebraska',
    'Vanderbilt Commodores': 'Vanderbilt',
    'North Carolina Tar Heels': 'North Carolina',
    "Saint Mary's Gaels": "Saint Mary's",
    'Clemson Tigers': 'Clemson',
    'Iowa Hawkeyes': 'Iowa',
    'Texas A&M Aggies': 'Texas A&M',
    'VCU Rams': 'VCU',
    'McNeese Cowboys': 'McNeese',
    'Troy Trojans': 'Troy',
    'Pennsylvania Quakers': 'Penn',
    'Idaho Vandals': 'Idaho',
    'Prairie View A&M Panthers': 'Prairie View A&M',
    'NC State Wolfpack': 'NC State',
    'Lehigh Mountain Hawks': 'Lehigh',
}


def _sn(full_name: str) -> str:
    """Return school name only (no mascot)."""
    return _SHORT_NAME_MAP.get(full_name, full_name.split()[0])


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


def create_bracket_figure(sim_results: dict) -> go.Figure:
    """
    Approach 2: Plotly scatter-based interactive bracket grid.

    Layout (11 x columns):
      col:  0    1    2    3    4   4.8  5.2   6    7    8    9    10
           R64  R32  S16   E8   FF  [L] [R]   FF   E8  S16  R32  R64
           ←── East + South (left) ────→   ←── West + Midwest (right) ──→

    Each blob = predicted winner of that bracket slot, coloured by
    overall championship win probability (green = favourite).
    Hover shows full round-by-round breakdown.
    """
    NCAA_SLOT_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]

    # Probability key used to determine who wins each round transition
    ROUND_ADV_KEYS = [
        'round_32_prob',    # R64 → R32
        'sweet_16_prob',    # R32 → S16
        'elite_8_prob',     # S16 → E8
        'final_four_prob',  # E8  → FF (regional champ)
        'championship_prob',# FF  → Finalist
    ]

    def region_slots(region_name):
        """Return [(team_id, stats)] in NCAA slot order for one region (16 slots)."""
        by_seed = {
            stats['team'].seed: (tid, stats)
            for tid, stats in sim_results.items()
            if stats['team'].region == region_name
        }
        return [by_seed.get(s) for s in NCAA_SLOT_ORDER]

    # ── y positions ──────────────────────────────────────────────────────────
    # 16 slots per region, 3-unit spacing, 6-unit gap between the two regions
    # on each side.  Both sides share identical y coords.
    SLOT_SPACING = 3
    REGION_GAP   = 6
    TOP_YS    = [i * SLOT_SPACING for i in range(16)]              # 0..45
    BOTTOM_YS = [TOP_YS[-1] + REGION_GAP + SLOT_SPACING + i * SLOT_SPACING
                 for i in range(16)]                                # 54..99

    def build_side(top_region, bottom_region):
        """Build initial [(entry, y)] list for 32 teams on one side.
        Both regions: seed 1 at lowest y index (visual top with reversed axis)."""
        top    = [(e, y) for e, y in zip(region_slots(top_region),    TOP_YS)]
        bottom = [(e, y) for e, y in zip(region_slots(bottom_region), BOTTOM_YS)]
        return top + bottom

    left_r64  = build_side('East',  'South')
    right_r64 = build_side('West',  'Midwest')

    def advance(prev_round, adv_key):
        """Pair adjacent entries; advance likely winner (higher adv_key); y = midpoint."""
        result = []
        for i in range(0, len(prev_round), 2):
            ea, ya = prev_round[i]
            eb, yb = prev_round[i + 1] if i + 1 < len(prev_round) else (None, ya)
            y_mid  = (ya + yb) / 2
            if ea is None and eb is None:
                result.append((None, y_mid))
            elif ea is None:
                result.append((eb, y_mid))
            elif eb is None:
                result.append((ea, y_mid))
            else:
                pa = ea[1].get(adv_key, 0)
                pb = eb[1].get(adv_key, 0)
                result.append((ea if pa >= pb else eb, y_mid))
        return result

    left_sched  = [left_r64]
    right_sched = [right_r64]
    for key in ROUND_ADV_KEYS:
        left_sched.append(advance(left_sched[-1],  key))
        right_sched.append(advance(right_sched[-1], key))

    # left_sched[5]  = [(left_finalist,  y)]   — 1 entry
    # right_sched[5] = [(right_finalist, y)]   — 1 entry

    # x positions: left side 0→4 then finalist at 4.2; right side 10→6 then 5.8
    # Finalists spread wider so champion at x=5, y above headers doesn't overlap
    LEFT_X  = [0, 1, 2, 3, 4, 4.2]
    RIGHT_X = [10, 9, 8, 7, 6, 5.8]

    fig = go.Figure()

    def _color(winner_prob: float) -> str:
        """Green gradient: grey at 0% → bright green at ≥5% championship prob."""
        t = min(1.0, winner_prob / 0.05)
        r = int(190 * (1 - t) + 30  * t)
        g = int(190 * (1 - t) + 200 * t)
        b = int(190 * (1 - t) + 60  * t)
        return f'rgb({r},{g},{b})'

    def _hover(name, team, stats):
        cp = stats.get('winner_prob', 0)
        return (
            f"<b>({team.seed}) {name}</b><br>"
            f"Region: {team.region}<br>"
            f"─────────────────<br>"
            f"Reach R32:  {stats.get('round_32_prob',    0):.1%}<br>"
            f"Reach S16:  {stats.get('sweet_16_prob',    0):.1%}<br>"
            f"Reach E8:   {stats.get('elite_8_prob',     0):.1%}<br>"
            f"Reach FF:   {stats.get('final_four_prob',  0):.1%}<br>"
            f"Reach Finals:{stats.get('championship_prob',0):.1%}<br>"
            f"<b>Win title: {cp:.1%}</b>"
        )

    for side_label, sched, x_list, text_side in [
        ('left',  left_sched,  LEFT_X,  'middle right'),
        ('right', right_sched, RIGHT_X, 'middle left'),
    ]:
        for round_idx, (entries, x) in enumerate(zip(sched, x_list)):
            marker_size = 12 + round_idx * 3       # markers grow each round
            text_size   = max(13, 13 + round_idx)

            xs, ys, lbls, colors, hovers = [], [], [], [], []
            for entry, y in entries:
                if entry is None:
                    continue
                tid, stats = entry
                t = stats['team']
                cp = stats.get('winner_prob', 0)
                xs.append(x);  ys.append(y)
                lbls.append(f"({t.seed}) {_sn(t.name)}")
                colors.append(_color(cp))
                hovers.append(_hover(_sn(t.name), t, stats))

            if not xs:
                continue
            fig.add_trace(go.Scatter(
                x=xs, y=ys,
                mode='markers+text',
                marker=dict(
                    size=marker_size,
                    color=colors,
                    symbol='square',
                    line=dict(width=1, color='#888'),
                ),
                text=lbls,
                textposition=text_side,
                textfont=dict(size=text_size, color='#222'),
                hovertext=hovers,
                hoverinfo='text',
                showlegend=False,
            ))

    # ── Championship special treatment ───────────────────────────────────────
    # Pick the predicted champion from the two finalists
    left_fin_entry,  left_fin_y  = left_sched[5][0]  if left_sched[5]  else (None, 48)
    right_fin_entry, right_fin_y = right_sched[5][0] if right_sched[5] else (None, 48)

    # Champion is shown as a dedicated row ABOVE the column headers (y=110)
    # so it never overlaps with the finalist squares below
    CHAMP_Y = 110

    champion_stats = None
    if left_fin_entry and right_fin_entry:
        lp = left_fin_entry[1].get('winner_prob', 0)
        rp = right_fin_entry[1].get('winner_prob', 0)
        champion_entry = left_fin_entry if lp >= rp else right_fin_entry
        _, champion_stats = champion_entry
    elif left_fin_entry:
        _, champion_stats = left_fin_entry
    elif right_fin_entry:
        _, champion_stats = right_fin_entry

    if champion_stats:
        ct = champion_stats['team']
        cp = champion_stats.get('winner_prob', 0)
        # Background highlight box
        fig.add_shape(
            type='rect', x0=3.8, y0=106.5, x1=6.2, y1=114.5,
            fillcolor='#fffde7', line=dict(color='gold', width=2), layer='below'
        )
        fig.add_trace(go.Scatter(
            x=[5], y=[CHAMP_Y],
            mode='markers+text',
            marker=dict(
                size=36, color=_color(cp), symbol='star',
                line=dict(width=2, color='gold'),
            ),
            text=[f"🏆 ({ct.seed}) {_sn(ct.name)}  {cp:.1%}"],
            textposition='middle right',
            textfont=dict(size=14, color='#7d6000', family='Arial Black'),
            hovertext=[_hover(_sn(ct.name), ct, champion_stats)],
            hoverinfo='text',
            showlegend=False,
        ))
        # Lines from finalists to champion box
        for fin_entry, fin_x in [(left_fin_entry, 4.2), (right_fin_entry, 5.8)]:
            if fin_entry:
                _, fin_stats = fin_entry
                champ_y_local = (left_fin_y + right_fin_y) / 2 if left_fin_entry and right_fin_entry else CHAMP_Y
                # find actual y for this finalist
                fin_y_actual = None
                for e, y in (left_sched[5] if fin_x < 5 else right_sched[5]):
                    if e == fin_entry:
                        fin_y_actual = y
                        break
                if fin_y_actual is not None:
                    fig.add_trace(go.Scatter(
                        x=[fin_x, 5], y=[fin_y_actual, CHAMP_Y],
                        mode='lines',
                        line=dict(color='#c8b400', width=1.5, dash='dot'),
                        showlegend=False, hoverinfo='skip'
                    ))

    # ── Round column headers ─────────────────────────────────────────────────
    header_cfg = [
        (0, 'R64'), (1, 'R32'), (2, 'S16'), (3, 'E8'), (4, 'Final Four'), (5, '🏆 Champ'),
        (6, 'Final Four'), (7, 'E8'), (8, 'S16'), (9, 'R32'), (10, 'R64'),
    ]
    for hx, hlabel in header_cfg:
        fig.add_annotation(x=hx, y=103, text=f'<b>{hlabel}</b>',
                           showarrow=False, font=dict(size=11, color='#555'),
                           xanchor='center')

    # ── Region labels ────────────────────────────────────────────────────────
    mid_top    = (TOP_YS[0]    + TOP_YS[-1])    / 2
    mid_bottom = (BOTTOM_YS[0] + BOTTOM_YS[-1]) / 2
    for lbl, lx, ly, color in [
        ('EAST',    -0.8, mid_top,    '#1a73e8'),
        ('SOUTH',   -0.8, mid_bottom, '#e67e22'),
        ('WEST',    10.8, mid_top,    '#27ae60'),
        ('MIDWEST', 10.8, mid_bottom, '#9b59b6'),
    ]:
        fig.add_annotation(x=lx, y=ly, text=f'<b>{lbl}</b>',
                           showarrow=False, font=dict(size=14, color=color),
                           xanchor='center', textangle=-90 if lx < 0 else 90)

    # ── Vertical column dividers ─────────────────────────────────────────────
    for col_x in [0, 1, 2, 3, 4, 4.2, 5.8, 6, 7, 8, 9, 10]:
        fig.add_shape(type='line', x0=col_x, y0=-2, x1=col_x, y1=101,
                      line=dict(color='#e8e8e8', width=1, dash='dot'), layer='below')

    # ── Region dividers (horizontal) ─────────────────────────────────────────
    div_y = (TOP_YS[-1] + BOTTOM_YS[0]) / 2
    fig.add_shape(type='line', x0=-0.5, y0=div_y, x1=10.5, y1=div_y,
                  line=dict(color='#cccccc', width=1.5), layer='below')

    fig.update_layout(
        title=dict(
            text='Interactive Bracket — Predicted Most-Likely Outcomes  '
                 '<span style="font-size:12px;color:#888">'
                 '(hover any square for full probability breakdown)</span>',
            font=dict(size=15),
        ),
        xaxis=dict(visible=False, range=[-1.2, 11.2]),
        yaxis=dict(visible=False, range=[-4, 116], autorange='reversed'),
        height=1400,
        margin=dict(l=0, r=0, t=60, b=5),
        plot_bgcolor='#f9f9f9',
        paper_bgcolor='white',
        hoverlabel=dict(bgcolor='white', font_size=13, bordercolor='#bbb'),
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
    
    # Reorder each region into the standard NCAA bracket slot order.
    # In a real bracket (top→bottom) the 16 first-round lines are:
    #   1, 16, 8, 9, 5, 12, 4, 13,  6, 11, 3, 14, 7, 10, 2, 15
    # This ensures consecutive slot-pairs (0,1),(2,3)... match the correct matchups:
    #   1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
    NCAA_SLOT_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    for region in regions:
        by_seed = {t[0]: t for t in regions[region]}
        ordered = []
        for slot_seed in NCAA_SLOT_ORDER:
            if slot_seed in by_seed:
                ordered.append(by_seed[slot_seed])
        # Append any remaining teams (shouldn't happen, but safety net)
        seen = {t[0] for t in ordered}
        for t in sorted(regions[region], key=lambda x: x[0]):
            if t[0] not in seen:
                ordered.append(t)
        regions[region] = ordered
    
    # Use module-level _sn as the short-name helper inside this function
    _short_name = _sn

    # Layout parameters for traditional bracket (pulled from LAYOUT_CONFIG)
    cfg = LAYOUT_CONFIG
    y_spacing = cfg.get('y_spacing', 32)
    x_round_spacing = cfg.get('x_round_spacing', 140)
    bracket_height = cfg.get('bracket_height', 16 * y_spacing)

    # Starting positions for each region (use the config block)
    region_layout = cfg.get('region_layout', {
        'East':    {'x_start': cfg.get('left_region_x', 50),    'y_start': 1020, 'y_direction': -1, 'direction': 1,  'label_pos': 'top-left'},
        'South':   {'x_start': cfg.get('left_region_x', 50),    'y_start': 510,  'y_direction': -1, 'direction': 1,  'label_pos': 'bottom-left'},
        'West':    {'x_start': cfg.get('right_region_x', 1825), 'y_start': 1020, 'y_direction': -1, 'direction': -1, 'label_pos': 'top-right'},
        'Midwest': {'x_start': cfg.get('right_region_x', 1825), 'y_start': 510,  'y_direction': -1, 'direction': -1, 'label_pos': 'bottom-right'},
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
        y_direction = layout.get('y_direction', 1)  # -1 = top region (downward), +1 = bottom region (upward)
        direction = layout['direction']
        
        # Add region label (use configurable offsets)
        label_x = x_start - cfg.get('label_outset', 25) if direction == 1 else x_start + cfg.get('label_outset', 25)
        label_angle = -90 if direction == 1 else 90
        fig.add_annotation(
            x=label_x, y=y_start + y_direction * bracket_height / 2,
            text=f"<b>{region_name.upper()}</b>",
            showarrow=False,
            font=dict(size=20, color='#2c3e50', family='Arial Black'),
            textangle=label_angle
        )
        
        # Round 1: First Four (all 16 teams)
        # Seed 1 anchored at y_start; bracket progresses in y_direction.
        # East/West (y_direction=-1): seed 1 at top (high y), bracket goes down.
        # South/Midwest (y_direction=+1): seed 1 at bottom (low y), bracket goes up.
        round1_positions = []
        for i, (seed, name, stats) in enumerate(teams):
            y_pos = y_start + i * y_direction * y_spacing
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
                text=_short_name(name),
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
                text=_short_name(winner[1]),
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
                text=_short_name(winner[1]),
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
                text=_short_name(winner[1]),
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
    # East is the top region (higher y), South is the bottom region (lower y)
    east_y  = left_semi_y + (y_spacing * 2)
    south_y = left_semi_y - (y_spacing * 2)
    
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
        text=_short_name(south_ff[1]),
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
        text=_short_name(east_ff[1]),
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
    # West is the top region (higher y), Midwest is the bottom region (lower y)
    west_y    = right_semi_y + (y_spacing * 2)
    midwest_y = right_semi_y - (y_spacing * 2)
    
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
        text=_short_name(midwest_ff[1]),
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
        text=_short_name(west_ff[1]),
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
        text=_short_name(left_winner[1]),
        showarrow=False,
        font=dict(size=14, color='#2c3e50', family='Arial'),
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
        text=_short_name(right_winner[1]),
        showarrow=False,
        font=dict(size=14, color='#2c3e50', family='Arial'),
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


def render_round_matchups(bracket_data_raw: dict, sim_results: dict):
    """Render round-by-round expected matchup analysis with spreads and totals."""
    import json
    from pathlib import Path

    REGIONS = ['East', 'South', 'Midwest', 'West']

    # ── efficiency-based spread / total estimator ──────────────────────────
    stats_lkp = {t['name']: t.get('stats', {}) for t in bracket_data_raw.get('teams', [])}

    def norm_cdf(x):
        """Standard normal CDF via math.erf (no scipy needed)."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def ats_side(prob_a, spread_raw):
        """Return ATS lean: ('fav', name_side, delta) or None.
        Compares model win-prob against the spread-implied win-prob.
        sigma=10 pts is a standard CBB approximation (1 pt ~ 3% WP near 50%).
        Returns 'fav' (take the spread's favourite to cover) or
        'dog' (take the underdog to cover), plus the delta.
        """
        sigma = 10.0
        spread_abs = abs(spread_raw)
        if spread_abs < 0.5:
            return None  # pick-em
        implied_wp = norm_cdf(spread_abs / sigma)
        # prob for whichever side the spread says is favoured
        prob_spread_fav = prob_a if spread_raw >= 0 else (1.0 - prob_a)
        delta = prob_spread_fav - implied_wp
        if delta > 0.10:
            return 'fav', delta
        elif delta < -0.10:
            return 'dog', abs(delta)
        return None

    def eff_pred(name_a, name_b):
        sa = stats_lkp.get(name_a, {})
        sb = stats_lkp.get(name_b, {})
        ea = float(sa.get('net_efficiency') or 0)
        eb = float(sb.get('net_efficiency') or 0)
        oa = float(sa.get('off_efficiency') or 110)
        ob = float(sb.get('off_efficiency') or 110)
        ta = float(sa.get('tempo') or 70)
        tb = float(sb.get('tempo') or 70)
        spread = (ea - eb) * 0.55          # positive = a favored
        raw_total = (oa * tb + ob * ta) / 100
        total = raw_total if 120 < raw_total < 185 else 148.5
        return round(spread, 1), round(total, 1)

    # ── index sim_results by region → seed ──────────────────────────────────
    by_rs = {}
    for _, stats in sim_results.items():
        t = stats['team']
        by_rs.setdefault(t.region, {})[t.seed] = stats

    def best_from(seeds, region, prob_key):
        candidates = [by_rs.get(region, {}).get(s) for s in seeds
                      if by_rs.get(region, {}).get(s)]
        return max(candidates, key=lambda s: s.get(prob_key, 0)) if candidates else None

    # ── confidence colour ───────────────────────────────────────────────────
    def conf_color(p):
        if p >= 0.80: return '#27ae60'
        if p >= 0.65: return '#f39c12'
        return '#e74c3c'

    def conf_label(p):
        if p >= 0.80: return 'High'
        if p >= 0.65: return 'Medium'
        return 'Toss-up'

    # ── render one structured game row ──────────────────────────────────────
    def game_row(short_a, seed_a, prob_a, short_b, seed_b, prob_b,
                 spread_raw, total, region, note='', date_str='', show_ats=True,
                 ats_html_override=None, spread_label='Spread'):
        fav_a = prob_a >= prob_b
        fav_prob = max(prob_a, prob_b)
        und_prob = min(prob_a, prob_b)
        fav_name = short_a if fav_a else short_b
        dog_name = short_b if fav_a else short_a
        # spread string — always shown as "<favored> -X.X"
        spread_str = (f"{fav_name} -{abs(spread_raw):.1f}" if abs(spread_raw) >= 0.5
                      else "Pick 'em")
        color = conf_color(fav_prob)
        # ATS lean — only shown for rounds where spread/WP come from independent sources
        if ats_html_override is not None:
            ats_html = ats_html_override
        elif show_ats:
            lean = ats_side(prob_a, spread_raw)
            if lean is None:
                ats_html = "<span style='color:#aaa'>ATS: Even</span>"
            elif lean[0] == 'fav':
                ats_html = (f"<span style='color:#27ae60;font-weight:600'>"
                            f"ATS: Take {fav_name} ✓</span>")
            else:
                ats_html = (f"<span style='color:#e67e22;font-weight:600'>"
                            f"ATS: Take {dog_name} +{abs(spread_raw):.1f} ✓</span>")
        else:
            ats_html = ''

        c1, c2, c3, c4, c5 = st.columns([3, 1, 3, 2, 2])
        with c1:
            if fav_a:
                st.markdown(
                    f"<span style='font-size:15px;font-weight:700'>"
                    f"({seed_a}) {short_a}</span> "
                    f"<span style='color:{color};font-weight:700'>{prob_a:.0%}</span>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<span style='font-size:14px;color:#555'>"
                    f"({seed_a}) {short_a}</span> "
                    f"<span style='color:#888'>{prob_a:.0%}</span>",
                    unsafe_allow_html=True)
        with c2:
            st.markdown("<div style='text-align:center;color:#aaa;font-size:13px;"
                        "padding-top:2px'>vs</div>", unsafe_allow_html=True)
        with c3:
            if not fav_a:
                st.markdown(
                    f"<span style='font-size:15px;font-weight:700'>"
                    f"({seed_b}) {short_b}</span> "
                    f"<span style='color:{color};font-weight:700'>{prob_b:.0%}</span>",
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f"<span style='font-size:14px;color:#555'>"
                    f"({seed_b}) {short_b}</span> "
                    f"<span style='color:#888'>{prob_b:.0%}</span>",
                    unsafe_allow_html=True)
        with c4:
            st.markdown(
                f"<div style='font-size:12px;color:#555'>"
                f"<b>{spread_label}:</b> {spread_str}<br>"
                f"<b>O/U:</b> {total:.1f}<br>"
                f"{ats_html}</div>",
                unsafe_allow_html=True)
        with c5:
            st.markdown(
                f"<div style='font-size:12px'>"
                f"<span style='color:{color};font-weight:600'>"
                f"{conf_label(fav_prob)}</span><br>"
                f"<span style='color:#888'>{region}</span>"
                f"{('<br><span style=\"color:#e74c3c\">⚡ ' + note + '</span>') if note else ''}"
                f"</div>",
                unsafe_allow_html=True)
        st.markdown("<hr style='margin:4px 0;border-color:#eee'>", unsafe_allow_html=True)

    # ── load R64 / First Four predictions ──────────────────────────────────
    r1_games, ff_games = [], []
    pred_files = sorted(Path('data_files/precomputed_predictions').glob(
        'tournament_predictions_*.json'))
    if pred_files:
        with open(pred_files[-1]) as f:
            pdata = json.load(f)
        for g in pdata['games']:
            if g['round_label'] == '1st Round':
                r1_games.append(g)
            elif g['round_label'] == 'First Four':
                ff_games.append(g)

    # ── load live Vegas odds (The Odds API) ─────────────────────────────────
    _live_odds = {}
    try:
        from fetch_live_odds import fetch_live_odds as _fetch_odds
        from fetch_live_odds import normalize_team_name as _norm_odds
        _live_odds = _fetch_odds()
    except Exception:
        pass

    def _std(name):
        """Normalize + collapse 'State'/'St' so both spellings match."""
        return _norm_odds(name).replace(' State', ' St') if _live_odds else name

    # Build a lookup: (norm_home, norm_away) -> odds dict (home-perspective)
    _odds_lkp = {}
    if _live_odds:
        for k, v in _live_odds.items():
            parts = k.split(' vs ')
            if len(parts) != 2:
                continue
            kh, ka = _std(parts[0]), _std(parts[1])
            _odds_lkp[(kh, ka)] = (v, False)   # not flipped
            _odds_lkp[(ka, kh)] = (v, True)    # home/away reversed

    def find_game_odds(home_team, away_team):
        """Return (odds_dict, spread_raw, total) from Vegas or None."""
        if not _odds_lkp:
            return None
        hn, an = _std(home_team), _std(away_team)
        entry = _odds_lkp.get((hn, an))
        if entry is None:
            return None
        v, flipped = entry
        if flipped:
            home_sp = v.get('away_spread')
        else:
            home_sp = v.get('home_spread')
        total = v.get('total_line')
        if home_sp is None or total is None:
            return None
        # Convention: spread_raw > 0 means home (team A) is favored
        spread_raw = -home_sp
        return spread_raw, float(total)

    R32_PAIRS  = [([1,16],[8,9]), ([5,12],[4,13]), ([6,11],[3,14]), ([7,10],[2,15])]
    S16_PAIRS  = [([1,16,8,9],[5,12,4,13]), ([6,11,3,14],[7,10,2,15])]
    E8_PAIRS   = [([1,16,8,9,5,12,4,13],[6,11,3,14,7,10,2,15])]
    FF_SEMIS   = [('East','South'), ('West','Midwest')]

    ROUND_PROB = {
        'Round of 32': 'round_32_prob',
        'Sweet 16':    'sweet_16_prob',
        'Elite 8':     'elite_8_prob',
    }

    # ── TABS ────────────────────────────────────────────────────────────────
    tab_labels = ['First Four', 'Round of 64', 'Round of 32',
                  'Sweet 16', 'Elite 8', 'Final Four', 'Championship']
    tabs = st.tabs(tab_labels)

    # header row helper
    def section_header():
        h1, h2, h3, h4, h5 = st.columns([3,1,3,2,2])
        h1.markdown("<span style='font-size:11px;color:#aaa;font-weight:600'>TEAM</span>",
                    unsafe_allow_html=True)
        h3.markdown("<span style='font-size:11px;color:#aaa;font-weight:600'>TEAM</span>",
                    unsafe_allow_html=True)
        h4.markdown("<span style='font-size:11px;color:#aaa;font-weight:600'>LINE / O/U / ATS</span>",
                    unsafe_allow_html=True)
        h5.markdown("<span style='font-size:11px;color:#aaa;font-weight:600'>CONFIDENCE</span>",
                    unsafe_allow_html=True)

    # ── Historical R64 ATS cover rates (favorite = lower seed) ─────────────
    # Sourced from multi-decade tournament ATS research (e.g. Action Network,
    # ESPN analytics). Rate = fraction of games where lower seed covers.
    R64_HIST_ATS = {
        (1, 16): 0.62,  # 1-seeds cover ~62% — strong historical edge
        (2, 15): 0.57,  # 2-seeds slight cover edge
        (3, 14): 0.54,  # mild fav edge
        (4, 13): 0.50,  # coin flip ATS
        (5, 12): 0.46,  # 12-seeds beat spread more often — famous trend
        (6, 11): 0.51,  # coin flip
        (7, 10): 0.49,  # slight dog edge
        (8,  9): 0.50,  # true coin flip
    }

    def hist_ats_html(seed_a, seed_b, short_a, short_b):
        """Return ATS indicator based on historical seed-matchup cover rates."""
        fav_seed  = min(seed_a, seed_b)
        dog_seed  = max(seed_a, seed_b)
        fav_name  = short_a if seed_a == fav_seed else short_b
        dog_name  = short_b if seed_a == fav_seed else short_a
        rate      = R64_HIST_ATS.get((fav_seed, dog_seed), 0.50)
        dog_rate  = 1.0 - rate
        if rate >= 0.55:
            return (f"<span style='color:#27ae60;font-weight:600'>"
                    f"ATS: {fav_name} covers hist ({rate:.0%})</span>")
        elif dog_rate >= 0.55:
            return (f"<span style='color:#e67e22;font-weight:600'>"
                    f"ATS: {dog_name} covers hist ({dog_rate:.0%})</span>")
        else:
            return "<span style='color:#aaa'>ATS: No hist. edge</span>"

    # ── First Four ──────────────────────────────────────────────────────────
    with tabs[0]:
        st.caption("Play-in games — winners advance to Round of 64")
        section_header()
        if ff_games:
            for g in sorted(ff_games, key=lambda x: x['region']):
                hw, aw = g['home_win_prob'], g['away_win_prob']
                sp = g['predicted_spread']
                ha = hist_ats_html(g['home_seed'], g['away_seed'],
                                   _sn(g['home_team']), _sn(g['away_team']))
                game_row(_sn(g['home_team']), g['home_seed'], hw,
                         _sn(g['away_team']), g['away_seed'], aw,
                         sp, g['predicted_total'], g['region'],
                         show_ats=False, ats_html_override=ha)
        else:
            st.info("No First Four predictions available.")

    # ── Round of 64 ─────────────────────────────────────────────────────────
    with tabs[1]:
        st.caption("Model predictions with spread and over/under")
        for region in REGIONS:
            st.markdown(f"#### {region}")
            section_header()
            rg = sorted([g for g in r1_games if g['region'] == region],
                        key=lambda x: min(x['home_seed'], x['away_seed']))
            for g in rg:
                hw, aw = g['home_win_prob'], g['away_win_prob']
                upset = g.get('upset_signal', False)
                note = ''
                if upset:
                    und = _sn(g['away_team']) if aw > hw else _sn(g['home_team'])
                    note = f"Upset alert: {und}"
                vegas = find_game_odds(g['home_team'], g['away_team'])
                if vegas:
                    v_spread, v_total = vegas
                    game_row(_sn(g['home_team']), g['home_seed'], hw,
                             _sn(g['away_team']), g['away_seed'], aw,
                             v_spread, v_total,
                             region, note, show_ats=True, spread_label='Vegas')
                else:
                    ha = hist_ats_html(g['home_seed'], g['away_seed'],
                                       _sn(g['home_team']), _sn(g['away_team']))
                    game_row(_sn(g['home_team']), g['home_seed'], hw,
                             _sn(g['away_team']), g['away_seed'], aw,
                             g['predicted_spread'], g['predicted_total'],
                             region, note, show_ats=False, ats_html_override=ha)

    # ── Round of 32 ─────────────────────────────────────────────────────────
    with tabs[2]:
        st.caption("Expected matchups based on Monte Carlo simulation")
        for region in REGIONS:
            st.markdown(f"#### {region}")
            section_header()
            for sa, sb in R32_PAIRS:
                ta = best_from(sa, region, 'round_32_prob')
                tb = best_from(sb, region, 'round_32_prob')
                if not ta or not tb: continue
                pa, pb = ta.get('round_32_prob', 0.5), tb.get('round_32_prob', 0.5)
                tot = pa + pb or 1
                cpa, cpb = pa/tot, pb/tot
                sp, ot = eff_pred(ta['team'].name, tb['team'].name)
                game_row(_sn(ta['team'].name), ta['team'].seed, cpa,
                         _sn(tb['team'].name), tb['team'].seed, cpb,
                         sp, ot, region)

    # ── Sweet 16 ────────────────────────────────────────────────────────────
    with tabs[3]:
        st.caption("Expected matchups — conditioned on R32 results")
        for region in REGIONS:
            st.markdown(f"#### {region}")
            section_header()
            for sa, sb in S16_PAIRS:
                ta = best_from(sa, region, 'sweet_16_prob')
                tb = best_from(sb, region, 'sweet_16_prob')
                if not ta or not tb: continue
                pa, pb = ta.get('sweet_16_prob', 0.5), tb.get('sweet_16_prob', 0.5)
                tot = pa + pb or 1
                cpa, cpb = pa/tot, pb/tot
                sp, ot = eff_pred(ta['team'].name, tb['team'].name)
                game_row(_sn(ta['team'].name), ta['team'].seed, cpa,
                         _sn(tb['team'].name), tb['team'].seed, cpb,
                         sp, ot, region)

    # ── Elite 8 ─────────────────────────────────────────────────────────────
    with tabs[4]:
        st.caption("Regional finals — winner goes to Final Four")
        for region in REGIONS:
            st.markdown(f"#### {region}")
            section_header()
            for sa, sb in E8_PAIRS:
                ta = best_from(sa, region, 'elite_8_prob')
                tb = best_from(sb, region, 'elite_8_prob')
                if not ta or not tb: continue
                pa, pb = ta.get('elite_8_prob', 0.5), tb.get('elite_8_prob', 0.5)
                tot = pa + pb or 1
                cpa, cpb = pa/tot, pb/tot
                sp, ot = eff_pred(ta['team'].name, tb['team'].name)
                game_row(_sn(ta['team'].name), ta['team'].seed, cpa,
                         _sn(tb['team'].name), tb['team'].seed, cpb,
                         sp, ot, region)

    # ── Final Four ──────────────────────────────────────────────────────────
    with tabs[5]:
        st.caption("National semifinals")
        section_header()
        for reg_a, reg_b in FF_SEMIS:
            reg_teams_a = [s for _, s in sim_results.items() if s['team'].region == reg_a]
            reg_teams_b = [s for _, s in sim_results.items() if s['team'].region == reg_b]
            ta = max(reg_teams_a, key=lambda s: s.get('final_four_prob', 0)) if reg_teams_a else None
            tb = max(reg_teams_b, key=lambda s: s.get('final_four_prob', 0)) if reg_teams_b else None
            if not ta or not tb: continue
            pa, pb = ta.get('final_four_prob', 0.5), tb.get('final_four_prob', 0.5)
            tot = pa + pb or 1
            cpa, cpb = pa/tot, pb/tot
            sp, ot = eff_pred(ta['team'].name, tb['team'].name)
            game_row(_sn(ta['team'].name), ta['team'].seed, cpa,
                     _sn(tb['team'].name), tb['team'].seed, cpb,
                     sp, ot, f"{reg_a} vs {reg_b}")

    # ── Championship ────────────────────────────────────────────────────────
    with tabs[6]:
        st.caption("National championship game")
        section_header()
        # Left finalist: East/South winner; Right finalist: West/Midwest winner
        left_teams  = [s for _, s in sim_results.items()
                       if s['team'].region in ('East', 'South')]
        right_teams = [s for _, s in sim_results.items()
                       if s['team'].region in ('West', 'Midwest')]
        ta = max(left_teams,  key=lambda s: s.get('championship_prob', 0)) if left_teams  else None
        tb = max(right_teams, key=lambda s: s.get('championship_prob', 0)) if right_teams else None
        if ta and tb:
            pa, pb = ta.get('championship_prob', 0.5), tb.get('championship_prob', 0.5)
            tot = pa + pb or 1
            cpa, cpb = pa/tot, pb/tot
            sp, ot = eff_pred(ta['team'].name, tb['team'].name)
            game_row(_sn(ta['team'].name), ta['team'].seed, cpa,
                     _sn(tb['team'].name), tb['team'].seed, cpb,
                     sp, ot, 'National Championship')
            st.markdown("")
            champ = ta if cpa >= cpb else tb
            st.success(f"🏆 **Predicted Champion: ({champ['team'].seed}) "
                       f"{_sn(champ['team'].name)}**  —  "
                       f"win probability {max(cpa,cpb):.0%}")


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
    
    # Show interactive grid bracket (Approach 2)
    if viz_mode in ["Interactive Grid", "All Views"]:
        st.header("🔲 Interactive Bracket Grid")
        grid_fig = create_bracket_figure(sim_results)
        st.plotly_chart(grid_fig)

        # Export controls (same as Visual Bracket)
        try:
            grid_png = pio.to_image(grid_fig, format='png', width=2400, height=1400, scale=2)
        except Exception:
            grid_png = None
        try:
            grid_svg = pio.to_image(grid_fig, format='svg', width=2400, height=1400, scale=1)
        except Exception:
            grid_svg = None
        try:
            grid_pdf = pio.to_image(grid_fig, format='pdf', width=2400, height=1400, scale=1)
        except Exception:
            grid_pdf = None

        col_exp, col_cap = st.columns([1, 4])
        with col_exp:
            if grid_png:
                st.download_button("Download PNG", data=grid_png, file_name="bracket_grid.png", mime="image/png", key="grid_png")
            if grid_svg:
                st.download_button("Download SVG", data=grid_svg, file_name="bracket_grid.svg", mime="image/svg+xml", key="grid_svg")
            if grid_pdf:
                st.download_button("Download PDF", data=grid_pdf, file_name="bracket_grid.pdf", mime="application/pdf", key="grid_pdf")
            if not grid_png and not grid_svg:
                st.info("Image export unavailable (install kaleido to enable)")
        with col_cap:
            st.caption(
                "💡 **How to read**: Each square is the predicted team for that bracket slot. "
                "Color = championship probability (green = title favourite). "
                "Hover any square for the full round-by-round breakdown. "
                "Star (⭐) = predicted champion."
            )
        st.divider()

    # Show matchup analysis
    if viz_mode in ["Matchup Analysis", "All Views"]:
        st.header("📋 Round-by-Round Matchup Analysis")
        render_round_matchups(bracket_data, sim_results)
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

    # ── Upset Watch ──────────────────────────────────────────────────────────
    with st.expander("🚨 Upset Watch & Cinderella Candidates", expanded=False):
        st.markdown(
            "Teams with meaningful upset potential based on efficiency metrics, "
            "seed matchups, and historical upset rates."
        )

        # Build a lightweight upset predictor (cached in session_state)
        if 'upset_predictor' not in st.session_state:
            with st.spinner("Training upset model…"):
                try:
                    _pred = UpsetPredictor()
                    _X_real, _y_real = create_training_data_from_csv()
                    _X_syn, _y_syn = create_historical_training_data()
                    if _X_real is not None and len(_X_real) >= 50:
                        _n = min(200, len(_X_syn))
                        _X = np.vstack([_X_real, _X_syn[:_n]])
                        _y = np.concatenate([_y_real, _y_syn[:_n]])
                    else:
                        _X, _y = _X_syn, _y_syn
                    _pred.train(_X, _y)
                    st.session_state['upset_predictor'] = _pred
                except Exception as _e:
                    st.session_state['upset_predictor'] = None

        upset_pred = st.session_state.get('upset_predictor')

        # Build efficiency lookup from bracket_data (name -> stats dict)
        _eff_lkp = {t['name']: t.get('stats', {}) or {}
                    for t in (bracket_data.get('teams', []) if isinstance(bracket_data, dict) else [])}

        # Build bracket_data structure expected by generate_upset_watch_list
        bracket_data_upset = []
        for team_id, stats in sim_results.items():
            team_obj = stats.get('team')
            if team_obj is None:
                continue
            seed = team_obj.seed or 8
            region = team_obj.region or 'Unknown'
            # Pair low seeds (underdogs) with their first-round opponent
            # (approximate: opponent seed = 17 - seed for standard bracket)
            opp_seed = max(1, 17 - seed)
            # Find opponent in sim_results by region + seed
            opp_stats = next(
                (s for _, s in sim_results.items()
                 if s.get('team')
                 and s['team'].region == region
                 and s['team'].seed == opp_seed),
                None
            )
            if opp_stats is None:
                continue
            opp_team = opp_stats['team']

            # helper to safely get stat values — use bracket_data efficiency lookup
            t_stats = _eff_lkp.get(team_obj.name, {})
            o_stats = _eff_lkp.get(opp_team.name, {})

            favorite_dict = {
                'seed': min(seed, opp_seed),
                'net_efficiency': (t_stats.get('net_efficiency', 10)
                                   if seed < opp_seed
                                   else o_stats.get('net_efficiency', 10)),
                'tempo': (t_stats.get('tempo', 70)
                          if seed < opp_seed
                          else o_stats.get('tempo', 70)),
                'three_rate': 0.35,
                'def_efficiency': 100,
            }
            underdog_dict = {
                'name': team_obj.name if seed > opp_seed else opp_team.name,
                'seed': max(seed, opp_seed),
                'net_efficiency': (t_stats.get('net_efficiency', 5)
                                   if seed > opp_seed
                                   else o_stats.get('net_efficiency', 5)),
                'tempo': (t_stats.get('tempo', 68)
                          if seed > opp_seed
                          else o_stats.get('tempo', 68)),
                'three_rate': 0.38,
                'def_efficiency': 100,
                'favorite_seed': min(seed, opp_seed),
                'favorite_name': (opp_team.name if seed > opp_seed else team_obj.name),
                'round': 1,
            }
            bracket_data_upset.append({'favorite': favorite_dict, 'underdog': underdog_dict})

        if upset_pred and bracket_data_upset:
            # Deduplicate matchups (each pair appears twice)
            seen = set()
            unique_matchups = []
            for m in bracket_data_upset:
                key = tuple(sorted([m['underdog']['name'], m['underdog']['favorite_name']]))
                if key not in seen:
                    seen.add(key)
                    unique_matchups.append(m)

            # generate_upset_watch_list expects {'first_round_games': [...]}
            # each game: {'favorite': {..., 'seed', 'name'}, 'underdog': {..., 'seed', 'name'}, 'round': ...}
            fmt_games = []
            for m in unique_matchups:
                fav = dict(m['favorite'])
                fav['name'] = m['underdog']['favorite_name']
                und = dict(m['underdog'])
                und['name'] = m['underdog']['name']
                fmt_games.append({'favorite': fav, 'underdog': und, 'round': 'Round of 64'})

            watch_list = generate_upset_watch_list({'first_round_games': fmt_games}, upset_pred)

            if watch_list:
                rows = []
                for entry in watch_list:
                    # Parse 'matchup': "(seed) name vs (seed) name"
                    matchup = entry.get('matchup', '')
                    rows.append({
                        'Matchup': matchup,
                        'Upset Prob': f"{entry.get('upset_probability', 0):.1%}",
                        'Historical Rate': f"{entry.get('historical_rate', 0):.1%}",
                        'Confidence': entry.get('confidence', 'unknown').title(),
                        'Key Factors': '; '.join(entry.get('key_reasons', [])[:2]) or '—',
                    })
                upset_df = pd.DataFrame(rows)
                st.dataframe(upset_df, width="stretch", hide_index=True)
            else:
                st.info("No high-probability upsets detected for the first round.")
        elif not upset_pred:
            st.warning("Upset model could not be loaded.")
        else:
            st.info("No matchup data available.")

        # ── Cinderella candidates from simulation results ─────────────────
        st.markdown("#### 🦋 Cinderella Candidates")
        st.caption("Seeds 10–16 with notable deep-run probability")

        cinderellas = [
            (tid, s) for tid, s in sim_results.items()
            if s.get('team') and s['team'].seed is not None
            and s['team'].seed >= 10
            and s.get('sweet_sixteen_prob', s.get('round_32_prob', 0)) > 0.08
        ]
        cinderellas.sort(
            key=lambda x: x[1].get('sweet_sixteen_prob', x[1].get('round_32_prob', 0)),
            reverse=True
        )

        if cinderellas:
            c_rows = []
            for _, cs in cinderellas[:10]:
                t = cs['team']
                fav_seed = 17 - t.seed
                seed_pair = (fav_seed, t.seed)
                hist_rate = HISTORICAL_UPSET_RATES.get(seed_pair, 0.0)
                c_rows.append({
                    'Team': t.name,
                    'Seed': t.seed,
                    'Region': t.region,
                    'R32 Prob': f"{cs.get('round_32_prob', 0):.1%}",
                    'S16 Prob': f"{cs.get('sweet_sixteen_prob', 0):.1%}",
                    'E8 Prob': f"{cs.get('elite_eight_prob', 0):.1%}",
                    'Hist. Upset Rate': f"{hist_rate:.0%}",
                })
            st.dataframe(pd.DataFrame(c_rows), width="stretch", hide_index=True)
        else:
            st.info("No notable Cinderella candidates detected.")

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
