# Roadmap: Bracket Visualization

*Interactive bracket display for March Madness predictions.*

## Visualization Goals

1. **Display full 64-team bracket** with predictions
2. **Show probability heatmaps** for each team's advancement
3. **Interactive exploration** - click teams for details
4. **Mobile-friendly** responsive design
5. **Real-time updates** as tournament progresses

## Technology Options

| Technology | Pros | Cons | Best For |
|------------|------|------|----------|
| **Streamlit** | Already using, easy | Limited interactivity | Quick MVP |
| **Plotly** | Interactive, in Streamlit | Learning curve | Charts/heatmaps |
| **D3.js** | Full control, beautiful | Complex, JS required | Custom bracket |
| **React + SVG** | Modern, reusable | Separate app needed | Production app |
| **HTML/CSS/JS** | Simple, portable | Manual layout | Static bracket |

## Approach 1: Streamlit Bracket (MVP)

```python
import streamlit as st
import pandas as pd

def render_bracket_streamlit(bracket_data: dict):
    """Render bracket using Streamlit columns."""
    
    st.title("🏀 March Madness Bracket")
    
    # Create 7 columns: R64, R32, S16, E8, FF, E8, S16, R32, R64
    # (symmetric bracket layout)
    
    regions = ['East', 'West', 'South', 'Midwest']
    
    # Top half: East and West
    st.header("East vs West")
    render_half_bracket(bracket_data, 'East', 'West')
    
    st.divider()
    
    # Bottom half: South and Midwest
    st.header("South vs Midwest")
    render_half_bracket(bracket_data, 'South', 'Midwest')
    
    # Final Four and Championship
    st.divider()
    render_final_four(bracket_data)


def render_half_bracket(data: dict, region1: str, region2: str):
    """Render half of the bracket (2 regions meeting in Elite 8)."""
    
    col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 2])
    
    # Region 1 - Left side
    with col1:
        st.subheader(f"📍 {region1}")
        render_region_games(data, region1, 'Round of 64')
    
    with col2:
        st.caption("Round of 32")
        render_region_games(data, region1, 'Round of 32')
    
    # Sweet 16 / Elite 8 in center
    with col3:
        st.caption("Sweet 16 / Elite 8")
        render_region_games(data, region1, 'Sweet 16')
        st.divider()
        render_region_games(data, region2, 'Sweet 16')
    
    # Region 2 - Right side (reversed)
    with col4:
        st.caption("Round of 32")
        render_region_games(data, region2, 'Round of 32')
    
    with col5:
        st.subheader(f"📍 {region2}")
        render_region_games(data, region2, 'Round of 64')


def render_region_games(data: dict, region: str, round_name: str):
    """Render games for a specific region and round."""
    
    games = [g for g in data.get('predictions', []) 
             if g.get('region') == region and g.get('round') == round_name]
    
    for game in games:
        winner = game['winner']
        prob = game['win_probability']
        is_upset = game.get('is_upset', False)
        
        # Color coding
        if prob > 0.8:
            color = "🟢"  # High confidence
        elif prob > 0.6:
            color = "🟡"  # Medium confidence
        else:
            color = "🔴"  # Low confidence (upset territory)
        
        upset_marker = "⚠️" if is_upset else ""
        
        st.markdown(f"{color} **{game['winner_seed']}** {winner} {upset_marker}")


def render_final_four(data: dict):
    """Render Final Four and Championship."""
    
    st.header("🏆 Final Four & Championship")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.subheader("Final Four")
        ff_games = [g for g in data.get('predictions', []) 
                    if g.get('round') == 'Final Four']
        for game in ff_games:
            st.markdown(f"**{game['winner_seed']}** {game['winner']}")
    
    with col2:
        st.subheader("🏆 Champion")
        champ_game = [g for g in data.get('predictions', []) 
                      if g.get('round') == 'Championship']
        if champ_game:
            champion = champ_game[0]['winner']
            prob = champ_game[0]['win_probability']
            st.success(f"**{champion}** ({prob:.1%})")
    
    with col3:
        # Show championship probability distribution
        st.subheader("Win Probability")
        if data.get('championship_distribution'):
            top_5 = sorted(
                data['championship_distribution'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for team, prob in top_5:
                st.metric(team, f"{prob:.1%}")
```

## Approach 2: Plotly Bracket Visualization

```python
import plotly.graph_objects as go
import plotly.express as px

def create_bracket_figure(bracket_data: dict) -> go.Figure:
    """Create interactive bracket visualization with Plotly."""
    
    fig = go.Figure()
    
    # Bracket dimensions
    n_rounds = 6
    teams_per_region = 16
    
    # Position calculations
    def get_game_positions(round_num: int, region_idx: int):
        """Calculate x, y positions for games in each round."""
        # x position based on round (0-5 for left side, 6-11 for right)
        x = round_num if region_idx < 2 else (11 - round_num)
        
        # y positions spread out more in early rounds
        n_games = 2 ** (4 - round_num) if round_num < 5 else 1
        spacing = 64 / n_games
        y_positions = [spacing * (i + 0.5) for i in range(n_games)]
        
        # Adjust for region (top/bottom half)
        if region_idx in [1, 3]:
            y_positions = [y + 64 for y in y_positions]
        
        return x, y_positions
    
    # Draw games for each round
    for round_num in range(1, 7):
        games = [g for g in bracket_data.get('predictions', [])
                 if get_round_number(g['round']) == round_num]
        
        for game in games:
            # Determine position
            region_idx = ['East', 'West', 'South', 'Midwest'].index(
                game.get('region', 'East')
            )
            x, y_positions = get_game_positions(round_num, region_idx)
            
            # Add team box
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y_positions[0]],
                mode='markers+text',
                text=f"{game['winner_seed']} {game['winner']}",
                textposition='middle right',
                marker=dict(
                    size=20,
                    color=get_prob_color(game['win_probability']),
                    symbol='square'
                ),
                hovertemplate=(
                    f"<b>{game['winner']}</b><br>"
                    f"Seed: {game['winner_seed']}<br>"
                    f"Win Prob: {game['win_probability']:.1%}<br>"
                    f"<extra></extra>"
                )
            ))
    
    # Layout
    fig.update_layout(
        title="March Madness Bracket",
        showlegend=False,
        xaxis=dict(visible=False, range=[-1, 12]),
        yaxis=dict(visible=False, range=[0, 128]),
        height=800,
        width=1200
    )
    
    return fig


def get_round_number(round_name: str) -> int:
    """Convert round name to number."""
    return {
        'Round of 64': 1, 'Round of 32': 2,
        'Sweet 16': 3, 'Elite 8': 4,
        'Final Four': 5, 'Championship': 6
    }.get(round_name, 0)


def get_prob_color(prob: float) -> str:
    """Get color based on win probability."""
    if prob > 0.8:
        return 'green'
    elif prob > 0.6:
        return 'yellow'
    elif prob > 0.4:
        return 'orange'
    else:
        return 'red'


def create_probability_heatmap(simulation_results: dict) -> go.Figure:
    """Create heatmap showing advancement probabilities."""
    
    teams = simulation_results['team_probabilities']
    
    # Sort by championship probability
    sorted_teams = sorted(
        teams.items(),
        key=lambda x: x[1]['champion_prob'],
        reverse=True
    )[:32]  # Top 32 teams
    
    # Prepare data
    team_names = [f"({t[1]['seed']}) {t[1]['name']}" for t in sorted_teams]
    rounds = ['R32', 'S16', 'E8', 'FF', 'Finals', 'Champ']
    
    prob_matrix = []
    for team_id, probs in sorted_teams:
        prob_matrix.append([
            probs['round_32_prob'],
            probs['sweet_16_prob'],
            probs['elite_8_prob'],
            probs['final_four_prob'],
            probs['finals_prob'],
            probs['champion_prob']
        ])
    
    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=rounds,
        y=team_names,
        colorscale='YlOrRd',
        text=[[f"{p:.0%}" for p in row] for row in prob_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate=(
            "Team: %{y}<br>"
            "Round: %{x}<br>"
            "Probability: %{z:.1%}<extra></extra>"
        )
    ))
    
    fig.update_layout(
        title="Team Advancement Probabilities",
        xaxis_title="Tournament Round",
        yaxis_title="Team",
        height=800
    )
    
    return fig
```

## Approach 3: SVG Bracket Template

```python
def generate_svg_bracket(bracket_data: dict) -> str:
    """Generate SVG markup for bracket visualization."""
    
    svg_width = 1400
    svg_height = 900
    
    # Start SVG
    svg = f'''<svg viewBox="0 0 {svg_width} {svg_height}" 
                  xmlns="http://www.w3.org/2000/svg">
    <style>
        .team-box {{ fill: #f0f0f0; stroke: #333; stroke-width: 1; }}
        .team-text {{ font-family: Arial, sans-serif; font-size: 10px; }}
        .seed {{ font-weight: bold; }}
        .winner {{ fill: #e8f5e9; }}
        .upset {{ fill: #fff3e0; }}
        .round-label {{ font-size: 12px; font-weight: bold; fill: #666; }}
        .region-label {{ font-size: 14px; font-weight: bold; fill: #333; }}
    </style>
    '''
    
    # Round X positions
    round_x = {
        1: [50, 1250],      # R64 (left and right)
        2: [150, 1150],     # R32
        3: [250, 1050],     # S16
        4: [350, 950],      # E8
        5: [500, 800],      # FF
        6: [650]            # Championship
    }
    
    # Team box dimensions
    box_width = 90
    box_height = 20
    
    # Generate team boxes for each round
    for round_num in range(1, 7):
        games = get_games_for_round(bracket_data, round_num)
        
        for i, game in enumerate(games):
            x = round_x[round_num][0] if i < len(games) // 2 else round_x[round_num][-1]
            y = calculate_y_position(round_num, i, svg_height)
            
            # Determine styling
            is_upset = game.get('is_upset', False)
            box_class = 'team-box upset' if is_upset else 'team-box winner'
            
            # Add team box
            svg += f'''
            <rect x="{x}" y="{y}" width="{box_width}" height="{box_height}" 
                  class="{box_class}" rx="3"/>
            <text x="{x + 5}" y="{y + 14}" class="team-text">
                <tspan class="seed">{game['winner_seed']}</tspan>
                {game['winner'][:12]}
            </text>
            '''
            
            # Add probability indicator
            prob = game['win_probability']
            indicator_color = get_prob_color(prob)
            svg += f'''
            <circle cx="{x + box_width - 10}" cy="{y + 10}" r="5" 
                    fill="{indicator_color}" opacity="0.8"/>
            '''
    
    # Add region labels
    regions = ['EAST', 'WEST', 'SOUTH', 'MIDWEST']
    for i, region in enumerate(regions):
        x = 50 if i % 2 == 0 else 1250
        y = 30 if i < 2 else 480
        svg += f'''
        <text x="{x}" y="{y}" class="region-label">{region}</text>
        '''
    
    # Add round labels
    round_labels = ['R64', 'R32', 'Sweet 16', 'Elite 8', 'Final Four', 'Championship']
    for i, label in enumerate(round_labels):
        x = round_x[i + 1][0]
        svg += f'<text x="{x}" y="15" class="round-label">{label}</text>'
    
    # Close SVG
    svg += '</svg>'
    
    return svg


def calculate_y_position(round_num: int, game_idx: int, svg_height: int) -> float:
    """Calculate Y position for a game box."""
    # Number of games in this round
    games_in_round = 2 ** (6 - round_num)
    
    # Spacing between games
    spacing = (svg_height - 100) / games_in_round
    
    # Y position
    return 50 + (game_idx * spacing) + (spacing / 2) - 10


def get_games_for_round(bracket_data: dict, round_num: int) -> list:
    """Get games for a specific round number."""
    round_name = {
        1: 'Round of 64', 2: 'Round of 32', 3: 'Sweet 16',
        4: 'Elite 8', 5: 'Final Four', 6: 'Championship'
    }[round_num]
    
    return [g for g in bracket_data.get('predictions', [])
            if g.get('round') == round_name]
```

## Interactive Features

```python
def add_bracket_interactivity(bracket_data: dict):
    """Add interactive features to Streamlit bracket."""
    
    st.sidebar.header("Bracket Controls")
    
    # Team search
    all_teams = list(set(
        g['winner'] for g in bracket_data.get('predictions', [])
    ))
    selected_team = st.sidebar.selectbox(
        "Find a team:",
        [""] + sorted(all_teams)
    )
    
    if selected_team:
        show_team_path(bracket_data, selected_team)
    
    # Round filter
    selected_round = st.sidebar.selectbox(
        "Jump to round:",
        ['Full Bracket', 'Round of 64', 'Round of 32', 'Sweet 16',
         'Elite 8', 'Final Four', 'Championship']
    )
    
    # Probability threshold
    min_prob = st.sidebar.slider(
        "Minimum confidence to show:",
        0.0, 1.0, 0.0, 0.05
    )
    
    # Upset toggle
    show_upsets_only = st.sidebar.checkbox("Show only upset predictions")
    
    return {
        'selected_team': selected_team,
        'selected_round': selected_round,
        'min_probability': min_prob,
        'show_upsets_only': show_upsets_only
    }


def show_team_path(bracket_data: dict, team_name: str):
    """Show a specific team's predicted tournament path."""
    
    st.subheader(f"📍 {team_name}'s Predicted Path")
    
    team_games = [
        g for g in bracket_data.get('predictions', [])
        if g['winner'] == team_name or g.get('loser') == team_name
    ]
    
    for game in team_games:
        if game['winner'] == team_name:
            st.success(f"✅ **{game['round']}**: Beats {game['loser']} ({game['win_probability']:.1%})")
        else:
            st.error(f"❌ **{game['round']}**: Loses to {game['winner']}")
            break  # Team is eliminated


def show_probability_table(simulation_results: dict):
    """Show sortable table of all team probabilities."""
    
    st.subheader("📊 Full Probability Table")
    
    # Create DataFrame
    rows = []
    for team_id, probs in simulation_results['team_probabilities'].items():
        rows.append({
            'Team': probs['name'],
            'Seed': probs['seed'],
            'Region': probs['region'],
            'R32': f"{probs['round_32_prob']:.1%}",
            'S16': f"{probs['sweet_16_prob']:.1%}",
            'E8': f"{probs['elite_8_prob']:.1%}",
            'FF': f"{probs['final_four_prob']:.1%}",
            'Finals': f"{probs['finals_prob']:.1%}",
            'Champ': f"{probs['champion_prob']:.1%}"
        })
    
    df = pd.DataFrame(rows)
    
    # Sort options
    sort_by = st.selectbox(
        "Sort by:",
        ['Champ', 'FF', 'E8', 'S16', 'Seed']
    )
    
    # Display table
    st.dataframe(
        df.sort_values(sort_by, ascending=(sort_by == 'Seed')),
        hide_index=True,
        width='stretch'
    )
```

## Export Options

```python
def export_bracket_image(bracket_svg: str, filename: str = 'bracket.png'):
    """Export bracket as PNG image."""
    import cairosvg
    
    cairosvg.svg2png(bytestring=bracket_svg.encode(), write_to=filename)
    return filename


def export_bracket_pdf(bracket_data: dict, filename: str = 'bracket.pdf'):
    """Export bracket as printable PDF."""
    from reportlab.lib.pagesizes import landscape, letter
    from reportlab.pdfgen import canvas
    
    c = canvas.Canvas(filename, pagesize=landscape(letter))
    
    # Add title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, 550, "March Madness Bracket Predictions")
    
    # Add bracket content
    # (Simplified - would need full layout logic)
    
    c.save()
    return filename


def create_shareable_bracket(bracket_data: dict, user_name: str = ""):
    """Create shareable bracket link/embed."""
    
    import json
    import base64
    
    # Encode bracket data
    bracket_json = json.dumps(bracket_data)
    encoded = base64.b64encode(bracket_json.encode()).decode()
    
    # Create shareable URL (would need backend)
    share_url = f"https://bracket-oracle.com/share?data={encoded[:50]}..."
    
    return {
        'url': share_url,
        'embed_code': f'<iframe src="{share_url}" width="800" height="600"></iframe>'
    }
```

## Next Steps

1. Implement Streamlit bracket view (MVP)
2. Add Plotly heatmap for probabilities
3. Create SVG template for printable bracket
4. See `roadmap-implementation.md` for integration
