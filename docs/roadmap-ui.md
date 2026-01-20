# Roadmap: User Interface (Betting Focus)

*Streamlit UI for March Madness betting predictions.*

## Application Structure

```python
# predictions.py
import streamlit as st

st.set_page_config(
    page_title="March Madness Betting Predictions",
    page_icon="🏀",
    layout="wide"
)

st.title("🏀 March Madness Betting Predictions")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Dashboard", "Game Predictions", "Value Bets", "Bracket Builder", "Model Performance"]
)
```

## Page 1: Dashboard

```python
def render_dashboard():
    """Main dashboard with today's picks."""
    st.header("Today's Best Bets")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model ATS Record", "42-31 (57.5%)")
    with col2:
        st.metric("O/U Record", "38-35 (52.1%)")
    with col3:
        st.metric("Moneyline ROI", "+8.2%")
    with col4:
        st.metric("Value Bet ROI", "+15.3%")
    
    st.subheader("📊 Today's Predictions")
    
    for game in todays_games:
        render_game_card(game)
```

## Page 2: Game Predictions

```python
def render_game_predictions():
    """Individual game betting analysis."""
    st.header("Game Predictions")
    
    # Game selector
    col1, col2 = st.columns(2)
    with col1:
        team1 = st.selectbox("Team 1", teams_list, key="t1")
    with col2:
        team2 = st.selectbox("Team 2", teams_list, key="t2")
    
    if st.button("Analyze Matchup"):
        prediction = predict_game(team1, team2)
        
        # Display predictions
        st.subheader("🎯 Predictions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Moneyline")
            st.metric(f"{team1} Win Prob", f"{prediction['team1_prob']*100:.1f}%")
            st.metric(f"{team2} Win Prob", f"{prediction['team2_prob']*100:.1f}%")
            st.success(f"Pick: **{prediction['winner']}**")
        
        with col2:
            st.markdown("### Spread")
            st.metric("Predicted Margin", f"{prediction['margin']:+.1f}")
            st.metric("Current Line", f"{prediction['spread']:+.1f}")
            pick = team1 if prediction['margin'] > -prediction['spread'] else team2
            st.success(f"Pick: **{pick}** {prediction['spread']:+.1f}")
        
        with col3:
            st.markdown("### Over/Under")
            st.metric("Predicted Total", f"{prediction['total']:.1f}")
            st.metric("Current Line", f"{prediction['ou_line']:.1f}")
            pick = "OVER" if prediction['total'] > prediction['ou_line'] else "UNDER"
            st.success(f"Pick: **{pick}** {prediction['ou_line']}")
```

## Page 3: Value Bets

```python
def render_value_bets():
    """Display value betting opportunities."""
    st.header("💰 Value Bets")
    
    st.info("Value bets occur when model probability exceeds implied odds probability")
    
    # Threshold slider
    min_edge = st.slider("Minimum Edge %", 1, 20, 5) / 100
    
    value_bets = find_value_bets(predictions, lines, threshold=min_edge)
    
    if not value_bets:
        st.warning("No value bets found with current threshold")
    else:
        for bet in value_bets:
            with st.container():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.write(f"**{bet['team']}**")
                with col2:
                    st.metric("Model Prob", f"{bet['model_prob']*100:.1f}%")
                with col3:
                    st.metric("Implied Prob", f"{bet['implied_prob']*100:.1f}%")
                with col4:
                    st.metric("Edge", f"+{bet['edge']*100:.1f}%", 
                              delta=f"{bet['moneyline']:+}")
                
                st.divider()
```

## Page 4: Spread Analysis

```python
def render_spread_analysis():
    """Detailed spread betting analysis."""
    st.header("📈 Spread Analysis")
    
    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        round_filter = st.selectbox(
            "Tournament Round",
            ["All", "First Round", "Second Round", "Sweet 16", "Elite 8", "Final Four", "Championship"]
        )
    with col2:
        spread_range = st.slider("Spread Range", -20.0, 20.0, (-15.0, 15.0))
    
    # Display spread picks
    spread_picks = get_spread_picks(round_filter, spread_range)
    
    st.dataframe(
        spread_picks[["matchup", "spread", "predicted_margin", "edge", "pick"]],
        use_container_width=True
    )
    
    # Visualization
    import plotly.express as px
    
    fig = px.scatter(
        spread_picks,
        x="spread",
        y="predicted_margin",
        color="pick",
        hover_data=["matchup", "edge"],
        title="Predicted Margin vs Spread"
    )
    fig.add_shape(type="line", x0=-20, y0=-20, x1=20, y1=20, 
                  line=dict(color="gray", dash="dash"))
    st.plotly_chart(fig, use_container_width=True)
```

## Page 5: Over/Under Analysis

```python
def render_ou_analysis():
    """Over/under betting analysis."""
    st.header("📊 Over/Under Analysis")
    
    # Filters
    tempo_filter = st.selectbox(
        "Pace Profile",
        ["All Games", "High Tempo (150+)", "Low Tempo (<140)", "Pace Mismatch"]
    )
    
    ou_picks = get_ou_picks(tempo_filter)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔺 Best Overs")
        overs = ou_picks[ou_picks["pick"] == "OVER"].head(5)
        for _, game in overs.iterrows():
            st.write(f"**{game['matchup']}**")
            st.write(f"  Line: {game['line']} | Projected: {game['projected']:.1f}")
    
    with col2:
        st.subheader("🔻 Best Unders")
        unders = ou_picks[ou_picks["pick"] == "UNDER"].head(5)
        for _, game in unders.iterrows():
            st.write(f"**{game['matchup']}**")
            st.write(f"  Line: {game['line']} | Projected: {game['projected']:.1f}")
```

## Game Card Component

```python
def render_game_card(game: dict):
    """Render a single game prediction card."""
    with st.container():
        st.markdown(f"### {game['team1']} vs {game['team2']}")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            winner = game['team1'] if game['team1_prob'] > 0.5 else game['team2']
            st.metric("Winner", winner, f"{max(game['team1_prob'], game['team2_prob'])*100:.0f}%")
        
        with col2:
            spread_pick = game['team1'] if game['predicted_margin'] > -game['spread'] else game['team2']
            st.metric("Spread Pick", f"{spread_pick} {game['spread']:+.1f}")
        
        with col3:
            ou_pick = "OVER" if game['predicted_total'] > game['ou_line'] else "UNDER"
            st.metric("O/U Pick", f"{ou_pick} {game['ou_line']}")
        
        with col4:
            if game.get('value_bet'):
                st.metric("Value", f"+{game['edge']*100:.1f}%", delta="VALUE")
            else:
                st.metric("Value", "None", delta=None)
        
        st.divider()
```

## Model Performance Page

```python
def render_model_performance():
    """Display historical model performance."""
    st.header("📈 Model Performance")
    
    # Performance tabs
    tab1, tab2, tab3 = st.tabs(["ATS", "Over/Under", "Moneyline"])
    
    with tab1:
        st.subheader("Against the Spread Performance")
        st.metric("Overall ATS", "57.3%")
        
        # By round
        ats_by_round = get_ats_performance_by_round()
        st.bar_chart(ats_by_round)
    
    with tab2:
        st.subheader("Over/Under Performance")
        st.metric("Overall O/U", "53.1%")
    
    with tab3:
        st.subheader("Moneyline Performance")
        st.metric("ROI", "+8.2%")
```

## Dependencies

```
# Already in requirements.txt
streamlit

# Add these
plotly
pandas
```

## Next Steps
- See `roadmap-betting-models.md` for prediction logic
- See `roadmap-betting-features.md` for available metrics
