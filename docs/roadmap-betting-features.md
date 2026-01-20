# Roadmap: Betting Features

*Feature engineering for spread, over/under, and moneyline predictions.*

## Efficiency-Based Features

### Net Efficiency Differential
The most predictive feature for game outcomes.

```python
def calculate_efficiency_differential(team1_eff: dict, team2_eff: dict) -> dict:
    """Calculate efficiency differentials between two teams."""
    return {
        "off_eff_diff": team1_eff["adj_offense"] - team2_eff["adj_offense"],
        "def_eff_diff": team1_eff["adj_defense"] - team2_eff["adj_defense"],
        "net_eff_diff": (
            (team1_eff["adj_offense"] - team1_eff["adj_defense"]) -
            (team2_eff["adj_offense"] - team2_eff["adj_defense"])
        )
    }
```

### Tempo-Adjusted Projections
For over/under predictions.

```python
def project_game_total(team1_eff: dict, team2_eff: dict) -> float:
    """Project total points using tempo and efficiency.
    
    Formula: (Team1_OE + Team2_OE) / 2 * Avg_Tempo / 100
    """
    avg_tempo = (team1_eff["tempo"] + team2_eff["tempo"]) / 2
    avg_off_eff = (team1_eff["adj_offense"] + team2_eff["adj_offense"]) / 2
    avg_def_eff = (team1_eff["adj_defense"] + team2_eff["adj_defense"]) / 2
    
    # Expected points per 100 possessions
    expected_eff = (avg_off_eff + avg_def_eff) / 2
    
    # Scale to game possessions
    projected_total = expected_eff * (avg_tempo / 100) * 2
    
    return projected_total
```

## Spread Prediction Features

```python
def calculate_spread_features(team1: dict, team2: dict) -> dict:
    """Features for predicting point spread."""
    return {
        # Efficiency-based
        "net_rating_diff": team1["net_rating"] - team2["net_rating"],
        "off_rating_diff": team1["off_rating"] - team2["off_rating"],
        "def_rating_diff": team1["def_rating"] - team2["def_rating"],
        
        # Scoring-based
        "ppg_diff": team1["ppg"] - team2["ppg"],
        "opp_ppg_diff": team1["opp_ppg"] - team2["opp_ppg"],
        "margin_diff": team1["avg_margin"] - team2["avg_margin"],
        
        # Seed-based (tournament specific)
        "seed_diff": team1["seed"] - team2["seed"],
        
        # Four factors differentials
        "efg_diff": team1["efg_pct"] - team2["efg_pct"],
        "to_rate_diff": team1["to_rate"] - team2["to_rate"],
        "orb_diff": team1["orb_pct"] - team2["orb_pct"],
        "ft_rate_diff": team1["ft_rate"] - team2["ft_rate"],
    }
```

## Over/Under Features

```python
def calculate_total_features(team1: dict, team2: dict) -> dict:
    """Features for predicting game total."""
    return {
        # Pace
        "combined_tempo": team1["tempo"] + team2["tempo"],
        "avg_tempo": (team1["tempo"] + team2["tempo"]) / 2,
        
        # Scoring
        "combined_ppg": team1["ppg"] + team2["ppg"],
        "combined_opp_ppg": team1["opp_ppg"] + team2["opp_ppg"],
        
        # Efficiency
        "combined_off_eff": team1["off_rating"] + team2["off_rating"],
        "combined_def_eff": team1["def_rating"] + team2["def_rating"],
        
        # Shooting
        "combined_fg_pct": team1["fg_pct"] + team2["fg_pct"],
        "combined_3pt_pct": team1["three_pct"] + team2["three_pct"],
        
        # Projected total
        "projected_total": project_game_total(team1, team2),
    }
```

## Moneyline/Win Probability Features

```python
def calculate_win_probability_features(team1: dict, team2: dict) -> dict:
    """Features for win probability prediction."""
    return {
        # Core predictors
        "net_rating_diff": team1["net_rating"] - team2["net_rating"],
        "seed_diff": team1["seed"] - team2["seed"],
        "ranking_diff": team1["ranking"] - team2["ranking"],
        
        # Experience
        "tournament_exp_diff": team1["tourney_appearances"] - team2["tourney_appearances"],
        
        # Form
        "last_10_win_pct_diff": team1["last_10_pct"] - team2["last_10_pct"],
        
        # SOS-adjusted
        "sos_adjusted_margin": team1["margin"] * team1["sos"] - team2["margin"] * team2["sos"],
    }
```

## Value Bet Detection

```python
def calculate_implied_probability(moneyline: int) -> float:
    """Convert American odds to implied probability."""
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return abs(moneyline) / (abs(moneyline) + 100)

def find_value_bets(predictions: list, lines: list, threshold: float = 0.05):
    """Find games where model probability exceeds implied probability."""
    value_bets = []
    
    for pred, line in zip(predictions, lines):
        implied_prob = calculate_implied_probability(line["moneyline"])
        edge = pred["win_prob"] - implied_prob
        
        if edge > threshold:
            value_bets.append({
                "team": pred["team"],
                "model_prob": pred["win_prob"],
                "implied_prob": implied_prob,
                "edge": edge,
                "moneyline": line["moneyline"]
            })
    
    return sorted(value_bets, key=lambda x: -x["edge"])
```

## ATS (Against the Spread) Features

```python
def calculate_ats_features(team1: dict, team2: dict, spread: float) -> dict:
    """Features for predicting ATS outcomes."""
    predicted_margin = team1["predicted_margin"]
    
    return {
        # Core
        "predicted_margin": predicted_margin,
        "spread": spread,
        "spread_diff": predicted_margin - spread,
        
        # Historical ATS
        "team1_ats_pct": team1.get("ats_pct", 0.5),
        "team2_ats_pct": team2.get("ats_pct", 0.5),
        
        # Cover tendency
        "team1_avg_cover_margin": team1.get("avg_cover", 0),
        "team2_avg_cover_margin": team2.get("avg_cover", 0),
        
        # As favorite/underdog
        "fav_ats_pct": team1.get("favorite_ats_pct", 0.5),
        "dog_ats_pct": team2.get("underdog_ats_pct", 0.5),
    }
```

## Feature Summary

| Bet Type | Key Features | Target Variable |
|----------|--------------|-----------------|
| Moneyline | net_rating_diff, seed_diff | Win (0/1) |
| Spread | margin_diff, efficiency_diff | Cover (0/1) |
| Over/Under | tempo, combined_ppg, projected_total | Over (0/1) |
| Value | model_prob - implied_prob | Edge % |

## Next Steps
- See `roadmap-betting-models.md` for model training
- See `roadmap-ui.md` for displaying predictions
