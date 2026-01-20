# Roadmap: Feature Engineering

*Suggested features to derive from raw data.*

## Team Performance Features

### Win/Loss Metrics
```python
import pandas as pd

def calculate_win_metrics(games_df: pd.DataFrame, team: str):
    """Calculate win-related features for a team."""
    home_games = games_df[games_df["home_team"] == team]
    away_games = games_df[games_df["away_team"] == team]
    
    home_wins = (home_games["home_points"] > home_games["away_points"]).sum()
    away_wins = (away_games["away_points"] > away_games["home_points"]).sum()
    
    total_games = len(home_games) + len(away_games)
    total_wins = home_wins + away_wins
    
    return {
        "win_pct": total_wins / total_games if total_games > 0 else 0,
        "home_win_pct": home_wins / len(home_games) if len(home_games) > 0 else 0,
        "away_win_pct": away_wins / len(away_games) if len(away_games) > 0 else 0
    }
```

### Scoring Metrics
```python
def calculate_scoring_features(games_df: pd.DataFrame, team: str):
    """Calculate scoring-related features."""
    home = games_df[games_df["home_team"] == team]
    away = games_df[games_df["away_team"] == team]
    
    points_for = home["home_points"].sum() + away["away_points"].sum()
    points_against = home["away_points"].sum() + away["home_points"].sum()
    total_games = len(home) + len(away)
    
    return {
        "ppg": points_for / total_games if total_games > 0 else 0,
        "ppg_allowed": points_against / total_games if total_games > 0 else 0,
        "point_diff": (points_for - points_against) / total_games if total_games > 0 else 0
    }
```

## Strength of Schedule Features

```python
def calculate_sos(games_df: pd.DataFrame, ratings_df: pd.DataFrame, team: str):
    """Calculate strength of schedule."""
    home = games_df[games_df["home_team"] == team]["away_team"]
    away = games_df[games_df["away_team"] == team]["home_team"]
    opponents = pd.concat([home, away])
    
    opponent_ratings = ratings_df[ratings_df["team"].isin(opponents)]["rating"]
    return opponent_ratings.mean() if len(opponent_ratings) > 0 else 0
```

## Momentum Features

```python
def calculate_momentum(games_df: pd.DataFrame, team: str, last_n: int = 5):
    """Calculate recent performance momentum."""
    team_games = games_df[
        (games_df["home_team"] == team) | (games_df["away_team"] == team)
    ].sort_values("start_date").tail(last_n)
    
    wins = 0
    for _, game in team_games.iterrows():
        if game["home_team"] == team:
            if game["home_points"] > game["away_points"]:
                wins += 1
        else:
            if game["away_points"] > game["home_points"]:
                wins += 1
    
    return wins / last_n if last_n > 0 else 0
```

## Advanced Efficiency Features

```python
def extract_efficiency_features(advanced_stats: dict):
    """Extract key efficiency metrics from advanced stats."""
    return {
        "off_success_rate": advanced_stats.get("offense", {}).get("success_rate", 0),
        "def_success_rate": advanced_stats.get("defense", {}).get("success_rate", 0),
        "off_explosiveness": advanced_stats.get("offense", {}).get("explosiveness", 0),
        "def_explosiveness": advanced_stats.get("defense", {}).get("explosiveness", 0),
        "off_ppa": advanced_stats.get("offense", {}).get("ppa", 0),
        "def_ppa": advanced_stats.get("defense", {}).get("ppa", 0)
    }
```

## Matchup Features

```python
def calculate_matchup_features(team1_stats: dict, team2_stats: dict):
    """Calculate head-to-head matchup differentials."""
    return {
        "rating_diff": team1_stats.get("rating", 0) - team2_stats.get("rating", 0),
        "ppg_diff": team1_stats.get("ppg", 0) - team2_stats.get("ppg", 0),
        "off_eff_diff": team1_stats.get("off_ppa", 0) - team2_stats.get("off_ppa", 0),
        "def_eff_diff": team1_stats.get("def_ppa", 0) - team2_stats.get("def_ppa", 0),
        "talent_diff": team1_stats.get("talent", 0) - team2_stats.get("talent", 0)
    }
```

## Feature Summary Table

| Feature Category | Features | Source Data |
|-----------------|----------|-------------|
| Win Metrics | win_pct, home/away win_pct | Games |
| Scoring | ppg, ppg_allowed, point_diff | Games |
| Strength | sos, opponent_rating_avg | Games + Ratings |
| Momentum | last_5_win_pct, streak | Games |
| Efficiency | success_rate, explosiveness, ppa | Advanced Stats |
| Matchup | rating_diff, ppg_diff, talent_diff | Derived |

## Next Steps
- See `roadmap-modeling.md` for model suggestions
- See `roadmap-ui.md` for visualization ideas
