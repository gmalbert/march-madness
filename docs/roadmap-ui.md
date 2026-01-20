# Roadmap: User Interface

*Suggestions for building the Streamlit UI.*

## Application Structure

```python
# predictions.py (main app)
import streamlit as st

st.set_page_config(
    page_title="March Madness Predictions",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 March Madness Predictions")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "Team Analysis", "Matchup Predictor", "Bracket Builder", "Historical Data"]
)
```

## Page 1: Home Dashboard

```python
def render_home():
    """Main dashboard with key metrics."""
    st.header("Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Teams Analyzed", "68")
    with col2:
        st.metric("Games in Database", "5,432")
    with col3:
        st.metric("Model Accuracy", "72.3%")
    with col4:
        st.metric("Last Updated", "Jan 19, 2026")
    
    st.subheader("Top Ranked Teams")
    # Display rankings table
    st.dataframe(rankings_df.head(25))
```

## Page 2: Team Analysis

```python
def render_team_analysis():
    """Individual team deep dive."""
    st.header("Team Analysis")
    
    team = st.selectbox("Select Team", teams_list)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Season Stats")
        st.metric("Record", f"{wins}-{losses}")
        st.metric("PPG", f"{ppg:.1f}")
        st.metric("Rating", f"{rating:.2f}")
    
    with col2:
        st.subheader("Advanced Metrics")
        st.metric("Offensive Efficiency", f"{off_eff:.2f}")
        st.metric("Defensive Efficiency", f"{def_eff:.2f}")
    
    st.subheader("Performance Trend")
    st.line_chart(performance_over_time)
```

## Page 3: Matchup Predictor

```python
def render_matchup():
    """Head-to-head prediction tool."""
    st.header("Matchup Predictor")
    
    col1, col2 = st.columns(2)
    
    with col1:
        team1 = st.selectbox("Team 1", teams_list, key="team1")
    with col2:
        team2 = st.selectbox("Team 2", teams_list, key="team2")
    
    if st.button("Predict Winner"):
        prob1, prob2 = predict_matchup(team1, team2)
        
        st.subheader("Prediction")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(team1, f"{prob1*100:.1f}%")
            st.progress(prob1)
        with col2:
            st.metric(team2, f"{prob2*100:.1f}%")
            st.progress(prob2)
        
        winner = team1 if prob1 > prob2 else team2
        st.success(f"Predicted Winner: **{winner}**")
```

## Page 4: Bracket Builder

```python
def render_bracket():
    """Interactive bracket builder."""
    st.header("Bracket Builder")
    
    st.info("Build your bracket with model-assisted predictions")
    
    # Round selection
    round_name = st.selectbox(
        "Round",
        ["First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    )
    
    # Display matchups for selected round
    for matchup in matchups[round_name]:
        col1, col2, col3 = st.columns([2, 1, 2])
        
        with col1:
            st.write(matchup["team1"])
        with col2:
            st.write("vs")
        with col3:
            st.write(matchup["team2"])
        
        selected = st.radio(
            f"Pick winner",
            [matchup["team1"], matchup["team2"]],
            key=f"{matchup['id']}"
        )
```

## Visualization Components

```python
import plotly.express as px
import plotly.graph_objects as go

def plot_team_comparison(team1_stats, team2_stats):
    """Radar chart for team comparison."""
    categories = ["Offense", "Defense", "Rebounding", "Turnovers", "Experience"]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=team1_stats,
        theta=categories,
        fill='toself',
        name=team1_name
    ))
    fig.add_trace(go.Scatterpolar(
        r=team2_stats,
        theta=categories,
        fill='toself',
        name=team2_name
    ))
    
    st.plotly_chart(fig)

def plot_win_probability_gauge(prob):
    """Gauge chart for win probability."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        title={'text': "Win Probability"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "lightgreen"}
            ]
        }
    ))
    st.plotly_chart(fig)
```

## UI/UX Best Practices

1. **Responsive Layout**: Use `st.columns()` for side-by-side content
2. **Loading States**: Use `st.spinner()` for async operations
3. **Caching**: Use `@st.cache_data` for expensive computations
4. **Session State**: Persist user selections across reruns

```python
@st.cache_data(ttl=3600)
def load_data():
    """Cache data loading for 1 hour."""
    return fetch_all_data()

# Session state for bracket picks
if "bracket_picks" not in st.session_state:
    st.session_state.bracket_picks = {}
```

## Next Steps
- See `roadmap-modeling.md` for prediction models
- See `roadmap-features.md` for available metrics
