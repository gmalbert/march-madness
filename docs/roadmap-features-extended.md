# Roadmap: Additional Features

*Extended functionality suggestions for the application.*

## Feature Ideas

### 1. Historical Bracket Analysis
Analyze past tournament performance.

```python
def analyze_historical_brackets(year: int):
    """Analyze how model would have performed in past tournaments."""
    # Load historical tournament games
    tournament_games = load_tournament_games(year)
    
    correct = 0
    total = 0
    
    for game in tournament_games:
        predicted = predict_winner(game["team1"], game["team2"])
        actual = game["winner"]
        
        if predicted == actual:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return {
        "year": year,
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }
```

### 2. Upset Detector
Identify potential upset candidates.

```python
def find_upset_candidates(matchups: list, threshold: float = 0.35):
    """Find games where underdog has decent upset chance."""
    upsets = []
    
    for matchup in matchups:
        lower_seed = matchup["lower_seed"]
        higher_seed = matchup["higher_seed"]
        
        prob = predict_win_probability(lower_seed, higher_seed)
        
        if prob >= threshold:
            upsets.append({
                "underdog": lower_seed,
                "favorite": higher_seed,
                "upset_prob": prob,
                "seed_diff": matchup["seed_diff"]
            })
    
    return sorted(upsets, key=lambda x: -x["upset_prob"])
```

### 3. Cinderella Tracker
Track mid-major and low-seed teams with upset potential.

```python
def identify_cinderellas(teams_data: dict, min_seed: int = 10):
    """Identify potential Cinderella teams."""
    cinderellas = []
    
    for team, data in teams_data.items():
        if data["seed"] >= min_seed:
            score = calculate_cinderella_score(data)
            if score > 0.5:
                cinderellas.append({
                    "team": team,
                    "seed": data["seed"],
                    "score": score,
                    "strengths": identify_strengths(data)
                })
    
    return sorted(cinderellas, key=lambda x: -x["score"])

def calculate_cinderella_score(data: dict) -> float:
    """Score a team's Cinderella potential."""
    factors = {
        "strong_defense": data.get("def_rank", 100) < 50,
        "experienced": data.get("avg_experience", 0) > 2.0,
        "good_coaching": data.get("coach_tournament_exp", 0) > 3,
        "hot_streak": data.get("last_10_win_pct", 0) > 0.7,
        "underrated": data.get("rating") > data.get("expected_rating", 0)
    }
    return sum(factors.values()) / len(factors)
```

### 4. Live Game Tracker
Monitor games in real-time during the tournament.

```python
def render_live_tracker():
    """Display live game information."""
    st.header("Live Games")
    
    live_games = fetch_live_games()
    
    for game in live_games:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])
            
            with col1:
                st.write(f"**{game['team1']}**")
                st.write(f"Score: {game['score1']}")
            
            with col2:
                st.write(f"Q{game['quarter']}")
                st.write(f"{game['time_remaining']}")
            
            with col3:
                st.write(f"**{game['team2']}**")
                st.write(f"Score: {game['score2']}")
            
            # Win probability chart
            st.progress(game['win_prob_team1'])
```

### 5. Bracket Comparison
Compare your bracket against the model.

```python
def compare_brackets(user_picks: dict, model_picks: dict):
    """Compare user bracket to model predictions."""
    comparison = {
        "agreements": 0,
        "disagreements": [],
        "risk_score": 0
    }
    
    for game_id, user_pick in user_picks.items():
        model_pick = model_picks.get(game_id)
        
        if user_pick == model_pick:
            comparison["agreements"] += 1
        else:
            comparison["disagreements"].append({
                "game": game_id,
                "user": user_pick,
                "model": model_pick,
                "model_confidence": model_picks.get(f"{game_id}_prob", 0.5)
            })
    
    return comparison
```

### 6. Pool Optimizer
Optimize bracket for pool scoring.

```python
def optimize_for_pool(pool_size: int, scoring_system: dict):
    """Generate bracket optimized for winning a pool."""
    base_bracket = generate_base_predictions()
    
    # Adjust picks based on pool strategy
    if pool_size > 50:
        # Larger pools need more differentiation
        base_bracket = add_calculated_upsets(base_bracket, risk_level="medium")
    else:
        # Smaller pools, stick with favorites
        base_bracket = add_calculated_upsets(base_bracket, risk_level="low")
    
    return base_bracket

def add_calculated_upsets(bracket: dict, risk_level: str):
    """Add strategic upsets based on risk tolerance."""
    upset_thresholds = {
        "low": 0.45,
        "medium": 0.35,
        "high": 0.25
    }
    threshold = upset_thresholds.get(risk_level, 0.35)
    
    # Identify high-value upset opportunities
    for game in bracket["games"]:
        if game["underdog_prob"] >= threshold:
            if game["round_multiplier"] * game["underdog_prob"] > 0.5:
                game["pick"] = game["underdog"]
    
    return bracket
```

### 7. Export Features

```python
def export_bracket_pdf(bracket: dict):
    """Export bracket to PDF."""
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    # ... render bracket
    c.save()
    
    st.download_button(
        "Download PDF",
        buffer.getvalue(),
        file_name="bracket.pdf",
        mime="application/pdf"
    )

def export_picks_csv(picks: dict):
    """Export picks to CSV."""
    df = pd.DataFrame(picks)
    csv = df.to_csv(index=False)
    
    st.download_button(
        "Download CSV",
        csv,
        file_name="picks.csv",
        mime="text/csv"
    )
```

## Feature Priority Matrix

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Matchup Predictor | Low | High | P0 |
| Team Analysis | Low | High | P0 |
| Bracket Builder | Medium | High | P1 |
| Upset Detector | Low | Medium | P1 |
| Historical Analysis | Medium | Medium | P2 |
| Cinderella Tracker | Low | Medium | P2 |
| Pool Optimizer | High | Medium | P3 |
| Live Tracker | High | Low | P3 |

## Dependencies to Add

```
# Add to requirements.txt
plotly
reportlab
xgboost
scikit-learn
pandas
numpy
```

## Next Steps
- Start with P0 features
- See `roadmap-ui.md` for UI implementation
- See `roadmap-modeling.md` for prediction logic
