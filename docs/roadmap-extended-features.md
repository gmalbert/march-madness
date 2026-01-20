# Roadmap: Additional Betting Features

*Extended functionality for March Madness betting predictions.*

## Upset Detector

```python
def find_upset_candidates(matchups: list, min_seed_diff: int = 4) -> list:
    """Identify games where lower seed has strong upset potential."""
    upsets = []
    
    for matchup in matchups:
        seed_diff = matchup["higher_seed"] - matchup["lower_seed"]
        
        if seed_diff >= min_seed_diff:
            # Calculate upset probability from efficiency
            upset_prob = predict_win_probability(
                matchup["lower_seed_team"],
                matchup["higher_seed_team"]
            )
            
            # Compare to implied probability from moneyline
            implied_prob = calculate_implied_probability(matchup["underdog_ml"])
            edge = upset_prob - implied_prob
            
            if upset_prob > 0.30:  # At least 30% chance
                upsets.append({
                    "underdog": matchup["lower_seed_team"],
                    "favorite": matchup["higher_seed_team"],
                    "seed_matchup": f"{matchup['lower_seed']} vs {matchup['higher_seed']}",
                    "upset_prob": upset_prob,
                    "implied_prob": implied_prob,
                    "edge": edge,
                    "moneyline": matchup["underdog_ml"]
                })
    
    return sorted(upsets, key=lambda x: -x["edge"])
```

## Live Line Movement Tracker

```python
def track_line_movement(game_id: str) -> dict:
    """Track how lines have moved since opening."""
    # Fetch current and opening lines
    current = get_current_lines(game_id)
    opening = get_opening_lines(game_id)
    
    return {
        "game_id": game_id,
        "spread_open": opening["spread"],
        "spread_current": current["spread"],
        "spread_movement": current["spread"] - opening["spread"],
        "total_open": opening["over_under"],
        "total_current": current["over_under"],
        "total_movement": current["over_under"] - opening["over_under"],
        "sharp_money_indicator": analyze_sharp_action(opening, current)
    }

def analyze_sharp_action(opening: dict, current: dict) -> str:
    """Detect potential sharp money movement."""
    spread_move = current["spread"] - opening["spread"]
    
    if abs(spread_move) >= 2:
        direction = "favorite" if spread_move < 0 else "underdog"
        return f"Sharp action on {direction}"
    return "No significant movement"
```

## Parlay Builder

```python
def build_parlay(picks: list) -> dict:
    """Calculate parlay odds and expected value."""
    total_odds = 1.0
    
    for pick in picks:
        # Convert American odds to decimal
        if pick["odds"] > 0:
            decimal = 1 + (pick["odds"] / 100)
        else:
            decimal = 1 + (100 / abs(pick["odds"]))
        
        total_odds *= decimal
    
    # Convert back to American
    if total_odds >= 2:
        american_odds = (total_odds - 1) * 100
    else:
        american_odds = -100 / (total_odds - 1)
    
    # Calculate combined probability
    combined_prob = 1.0
    for pick in picks:
        combined_prob *= pick["model_prob"]
    
    # Expected value
    ev = (combined_prob * (total_odds - 1)) - (1 - combined_prob)
    
    return {
        "picks": picks,
        "parlay_odds": american_odds,
        "decimal_odds": total_odds,
        "combined_prob": combined_prob,
        "expected_value": ev,
        "is_positive_ev": ev > 0
    }
```

## Historical ATS Trends

```python
def analyze_ats_trends(team: str, years: int = 5) -> dict:
    """Analyze team's historical against-the-spread performance."""
    games = fetch_team_games_with_lines(team, years)
    
    results = {
        "overall_ats": {"wins": 0, "losses": 0, "pushes": 0},
        "as_favorite": {"wins": 0, "losses": 0},
        "as_underdog": {"wins": 0, "losses": 0},
        "tournament_ats": {"wins": 0, "losses": 0},
        "by_spread_range": {}
    }
    
    for game in games:
        covered = game["margin"] > -game["spread"]
        
        # Overall
        if covered:
            results["overall_ats"]["wins"] += 1
        else:
            results["overall_ats"]["losses"] += 1
        
        # As favorite/underdog
        if game["spread"] < 0:  # Favorite
            key = "as_favorite"
        else:
            key = "as_underdog"
        
        if covered:
            results[key]["wins"] += 1
        else:
            results[key]["losses"] += 1
    
    return results
```

## Bracket Simulator

```python
import random

def simulate_bracket(predictions: dict, num_sims: int = 10000) -> dict:
    """Monte Carlo simulation of bracket outcomes."""
    results = {team: {"final_four": 0, "championship": 0, "winner": 0} 
               for team in predictions["teams"]}
    
    for _ in range(num_sims):
        bracket = run_single_simulation(predictions)
        
        for team in bracket["final_four"]:
            results[team]["final_four"] += 1
        for team in bracket["championship"]:
            results[team]["championship"] += 1
        results[bracket["winner"]]["winner"] += 1
    
    # Convert to percentages
    for team in results:
        results[team]["final_four_pct"] = results[team]["final_four"] / num_sims
        results[team]["championship_pct"] = results[team]["championship"] / num_sims
        results[team]["winner_pct"] = results[team]["winner"] / num_sims
    
    return results

def run_single_simulation(predictions: dict) -> dict:
    """Run a single bracket simulation using probabilities."""
    bracket = predictions.copy()
    
    for round_name in ["first", "second", "sweet16", "elite8", "final4", "championship"]:
        matchups = bracket[round_name]
        winners = []
        
        for matchup in matchups:
            prob = matchup["team1_prob"]
            winner = matchup["team1"] if random.random() < prob else matchup["team2"]
            winners.append(winner)
        
        bracket[f"{round_name}_winners"] = winners
    
    return bracket
```

## ROI Tracker

```python
def track_betting_roi(bets: list, bankroll: float = 1000) -> dict:
    """Track betting performance and ROI over time."""
    history = []
    current_bankroll = bankroll
    
    for bet in bets:
        stake = bet.get("stake", 100)
        
        if bet["result"] == "win":
            if bet["odds"] > 0:
                profit = stake * (bet["odds"] / 100)
            else:
                profit = stake * (100 / abs(bet["odds"]))
            current_bankroll += profit
        elif bet["result"] == "loss":
            current_bankroll -= stake
        # Push = no change
        
        history.append({
            "date": bet["date"],
            "bankroll": current_bankroll,
            "bet": bet
        })
    
    return {
        "starting_bankroll": bankroll,
        "ending_bankroll": current_bankroll,
        "profit": current_bankroll - bankroll,
        "roi_pct": (current_bankroll - bankroll) / bankroll * 100,
        "history": history
    }
```

## Export Picks

```python
def export_picks_to_csv(picks: list, filename: str = "picks.csv"):
    """Export betting picks to CSV."""
    import pandas as pd
    
    df = pd.DataFrame(picks)
    df.to_csv(filename, index=False)
    
    return filename

def export_picks_to_json(picks: list, filename: str = "picks.json"):
    """Export betting picks to JSON."""
    import json
    
    with open(filename, "w") as f:
        json.dump(picks, f, indent=2, default=str)
    
    return filename
```

## Feature Priority for Betting

| Feature | Effort | Impact | Priority |
|---------|--------|--------|----------|
| Game Predictions | Medium | High | P0 |
| Value Bet Finder | Low | High | P0 |
| Spread Analysis | Medium | High | P0 |
| O/U Analysis | Medium | High | P0 |
| Upset Detector | Low | Medium | P1 |
| Model Performance | Medium | Medium | P1 |
| Line Movement | Medium | Medium | P2 |
| Parlay Builder | Low | Low | P2 |
| Bracket Simulator | High | Medium | P3 |

## Next Steps
- See `roadmap-betting-models.md` for model training
- See `roadmap-ui.md` for UI implementation
