# Roadmap: Implementation Plan

*Step-by-step plan to build the complete March Madness prediction system.*

## Phase Overview

```
Phase 1: Data Foundation (Week 1-2)
├── Historical tournament data collection
├── Bracket structure implementation
└── Data pipeline setup

Phase 2: Model Development (Week 2-3)
├── Tournament-specific model training
├── Upset prediction model
└── Model evaluation and tuning

Phase 3: Simulation Engine (Week 3-4)
├── Monte Carlo simulator
├── Bracket state management
└── Results aggregation

Phase 4: Visualization (Week 4-5)
├── Bracket UI in Streamlit
├── Probability heatmaps
└── Interactive features

Phase 5: Integration & Polish (Week 5-6)
├── End-to-end testing
├── Real-time updates
└── Deployment
```

## Phase 1: Data Foundation

### Task 1.1: Create Bracket Data Structures

```python
# File: bracket_structures.py

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

class Region(Enum):
    EAST = "East"
    WEST = "West"
    SOUTH = "South"
    MIDWEST = "Midwest"

class Round(Enum):
    FIRST_ROUND = "Round of 64"
    SECOND_ROUND = "Round of 32"
    SWEET_16 = "Sweet 16"
    ELITE_8 = "Elite 8"
    FINAL_FOUR = "Final Four"
    CHAMPIONSHIP = "Championship"

@dataclass
class TournamentTeam:
    """A team in the tournament."""
    id: str
    name: str
    seed: int
    region: Region
    
    # Season stats
    wins: int = 0
    losses: int = 0
    net_efficiency: float = 0.0
    off_efficiency: float = 0.0
    def_efficiency: float = 0.0
    tempo: float = 70.0
    
    # Advanced metrics
    kenpom_rank: Optional[int] = None
    ap_rank: Optional[int] = None
    sos_rank: Optional[int] = None
    
    # Tournament experience
    coach_ncaa_games: int = 0
    program_final_fours: int = 0
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'seed': self.seed,
            'region': self.region.value,
            'wins': self.wins,
            'losses': self.losses,
            'net_efficiency': self.net_efficiency,
            'off_efficiency': self.off_efficiency,
            'def_efficiency': self.def_efficiency,
            'tempo': self.tempo
        }

@dataclass
class TournamentGame:
    """A single tournament game."""
    id: str
    round: Round
    region: Optional[Region]  # None for Final Four/Championship
    
    team1: TournamentTeam
    team2: TournamentTeam
    
    # Prediction
    team1_win_prob: float = 0.5
    predicted_winner: Optional[TournamentTeam] = None
    
    # Actual result (filled in as tournament progresses)
    winner: Optional[TournamentTeam] = None
    team1_score: Optional[int] = None
    team2_score: Optional[int] = None
    
    def is_upset(self) -> bool:
        """Check if result was an upset."""
        if self.winner:
            return self.winner.seed > min(self.team1.seed, self.team2.seed)
        return False

@dataclass  
class Bracket:
    """Complete tournament bracket."""
    year: int
    teams: Dict[str, TournamentTeam] = field(default_factory=dict)
    games: Dict[str, TournamentGame] = field(default_factory=dict)
    
    def get_teams_by_region(self, region: Region) -> List[TournamentTeam]:
        """Get all teams in a region."""
        return [t for t in self.teams.values() if t.region == region]
    
    def get_games_by_round(self, round: Round) -> List[TournamentGame]:
        """Get all games in a round."""
        return [g for g in self.games.values() if g.round == round]
    
    def save(self, filepath: str):
        """Save bracket to JSON."""
        data = {
            'year': self.year,
            'teams': {k: v.to_dict() for k, v in self.teams.items()},
            'games': {}  # Would need game serialization
        }
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: str) -> 'Bracket':
        """Load bracket from JSON."""
        data = json.loads(Path(filepath).read_text())
        # Would need deserialization logic
        return cls(year=data['year'])
```

### Task 1.2: Historical Data Collection

```python
# File: tournament_data_collection.py

import pandas as pd
import requests
from pathlib import Path

DATA_DIR = Path("data_files/tournament")
DATA_DIR.mkdir(parents=True, exist_ok=True)

def fetch_historical_tournament_results(start_year: int = 2010, 
                                        end_year: int = 2024) -> pd.DataFrame:
    """
    Fetch historical tournament results.
    Sources: sports-reference.com, kaggle NCAA datasets
    """
    
    all_games = []
    
    for year in range(start_year, end_year + 1):
        # Skip 2020 (COVID cancellation)
        if year == 2020:
            continue
        
        # Fetch from ESPN historical data
        games = fetch_year_tournament(year)
        all_games.extend(games)
    
    df = pd.DataFrame(all_games)
    df.to_csv(DATA_DIR / "historical_tournament_games.csv", index=False)
    
    return df

def fetch_year_tournament(year: int) -> list:
    """Fetch tournament games for a single year."""
    
    # ESPN endpoint for historical tournament data
    # Note: This is simplified - actual implementation would need proper API handling
    
    games = []
    
    # Would iterate through rounds and fetch actual data
    # Placeholder structure:
    rounds = ['First Round', 'Second Round', 'Sweet 16', 
              'Elite 8', 'Final Four', 'Championship']
    
    for round_name in rounds:
        # Fetch games for this round
        # games.extend(fetch_round_games(year, round_name))
        pass
    
    return games

def create_training_dataset() -> pd.DataFrame:
    """
    Create training dataset from historical tournament games.
    """
    
    # Load historical games
    historical = pd.read_csv(DATA_DIR / "historical_tournament_games.csv")
    
    # Load corresponding team stats for each year
    training_rows = []
    
    for _, game in historical.iterrows():
        year = game['year']
        team1_id = game['team1_id']
        team2_id = game['team2_id']
        
        # Get team stats for that year
        team1_stats = get_historical_team_stats(team1_id, year)
        team2_stats = get_historical_team_stats(team2_id, year)
        
        if team1_stats and team2_stats:
            row = create_training_features(
                team1_stats, team2_stats,
                game['round'], game['winner_id'] == team1_id
            )
            training_rows.append(row)
    
    return pd.DataFrame(training_rows)

def get_historical_team_stats(team_id: str, year: int) -> dict:
    """Get team stats for a historical year."""
    # Would load from cached historical data
    return {}

def create_training_features(team1: dict, team2: dict, 
                            round_name: str, team1_won: bool) -> dict:
    """Create feature row for training."""
    
    return {
        'net_eff_diff': team1.get('net_efficiency', 0) - team2.get('net_efficiency', 0),
        'off_eff_diff': team1.get('off_efficiency', 0) - team2.get('off_efficiency', 0),
        'def_eff_diff': team1.get('def_efficiency', 0) - team2.get('def_efficiency', 0),
        'seed_diff': team1.get('seed', 8) - team2.get('seed', 8),
        'round': round_name,
        'target': 1 if team1_won else 0
    }
```

### Task 1.3: Fetch Official Bracket

```python
# File: fetch_bracket.py

import requests
from datetime import datetime
from typing import Optional

def fetch_current_bracket() -> dict:
    """
    Fetch the current NCAA tournament bracket.
    Call this after Selection Sunday.
    """
    
    year = datetime.now().year
    
    # ESPN bracket API
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    
    params = {
        'dates': f'{year}0310-{year}0410',
        'groups': '100',  # Tournament games
        'limit': 100
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    return parse_espn_bracket(data)

def parse_espn_bracket(data: dict) -> dict:
    """Parse ESPN response into bracket structure."""
    
    bracket = {
        'year': data.get('season', {}).get('year'),
        'regions': {
            'East': [],
            'West': [],
            'South': [],
            'Midwest': []
        },
        'games': []
    }
    
    for event in data.get('events', []):
        game = parse_game(event)
        if game:
            bracket['games'].append(game)
    
    return bracket

def parse_game(event: dict) -> Optional[dict]:
    """Parse a single game from ESPN data."""
    
    competitions = event.get('competitions', [])
    if not competitions:
        return None
    
    competition = competitions[0]
    competitors = competition.get('competitors', [])
    
    if len(competitors) != 2:
        return None
    
    game = {
        'id': event['id'],
        'date': event['date'],
        'name': event.get('name', ''),
        'status': event.get('status', {}).get('type', {}).get('name'),
        'teams': []
    }
    
    for comp in competitors:
        team = {
            'id': comp['id'],
            'name': comp.get('team', {}).get('displayName'),
            'abbreviation': comp.get('team', {}).get('abbreviation'),
            'seed': comp.get('curatedRank', {}).get('current'),
            'score': comp.get('score'),
            'winner': comp.get('winner', False),
            'home_away': comp.get('homeAway')
        }
        game['teams'].append(team)
    
    return game
```

## Phase 2: Model Development

### Task 2.1: Train Tournament Model

```python
# File: train_tournament_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
import joblib
from pathlib import Path

MODEL_DIR = Path("data_files/models")

def train_tournament_predictor():
    """Train the main tournament prediction model."""
    
    # Load training data
    df = pd.read_csv("data_files/tournament/training_data.csv")
    
    # Features
    feature_cols = [
        'net_eff_diff', 'off_eff_diff', 'def_eff_diff',
        'seed_diff', 'coach_exp_diff', 'tempo_diff',
        'round_number'
    ]
    
    X = df[feature_cols]
    y = df['target']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train XGBoost
    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X, y, cv=5)
    
    print(f"Train Accuracy: {train_score:.3f}")
    print(f"Test Accuracy: {test_score:.3f}")
    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
    
    # Save model
    joblib.dump(model, MODEL_DIR / "tournament_predictor.joblib")
    
    # Feature importance
    importance = dict(zip(feature_cols, model.feature_importances_))
    print("\nFeature Importance:")
    for feat, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {feat}: {imp:.3f}")
    
    return model

def train_upset_model():
    """Train specialized upset prediction model."""
    
    # Load training data with upset labels
    df = pd.read_csv("data_files/tournament/training_data.csv")
    
    # Create upset target
    df['is_upset'] = (df['seed_diff'] > 0) & (df['target'] == 0)
    
    # Only use games where there's an upset possibility
    df_upset = df[df['seed_diff'] != 0]
    
    feature_cols = [
        'seed_diff', 'net_eff_diff', 'off_eff_diff',
        'tempo_diff', 'three_rate_underdog', 'last_10_diff'
    ]
    
    X = df_upset[feature_cols]
    y = df_upset['is_upset'].astype(int)
    
    # Train model optimized for upset detection
    model = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        min_samples_split=20,
        random_state=42
    )
    
    model.fit(X, y)
    
    # Save
    joblib.dump(model, MODEL_DIR / "upset_predictor.joblib")
    
    return model

if __name__ == "__main__":
    print("Training Tournament Predictor...")
    train_tournament_predictor()
    
    print("\nTraining Upset Predictor...")
    train_upset_model()
```

## Phase 3: Simulation Engine

### Task 3.1: Build Simulator

```python
# File: bracket_simulator.py

import numpy as np
from typing import Dict, List
from dataclasses import dataclass
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

from bracket_structures import Bracket, TournamentTeam, TournamentGame, Round, Region

@dataclass
class SimulationResult:
    """Result of a single bracket simulation."""
    champion: str
    final_four: List[str]
    round_winners: Dict[str, List[str]]

class BracketSimulator:
    """Monte Carlo bracket simulator."""
    
    def __init__(self, predictor, seed: int = None):
        self.predictor = predictor
        self.rng = np.random.default_rng(seed)
    
    def simulate_game(self, team1: TournamentTeam, team2: TournamentTeam,
                      round: Round) -> TournamentTeam:
        """Simulate a single game."""
        
        # Get win probability
        features = self._create_features(team1, team2, round)
        prob = self.predictor.predict_proba([features])[0][1]
        
        # Sample winner
        if self.rng.random() < prob:
            return team1
        return team2
    
    def simulate_tournament(self, bracket: Bracket) -> SimulationResult:
        """Simulate complete tournament once."""
        
        # Track remaining teams by region
        remaining = {
            region: bracket.get_teams_by_region(region)
            for region in Region
        }
        
        round_winners = {}
        
        # Simulate each round
        for round in Round:
            winners = self._simulate_round(remaining, round)
            round_winners[round.value] = [w.name for w in winners]
            
            # Update remaining teams
            if round.value in ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8']:
                for region in Region:
                    remaining[region] = [t for t in remaining[region] if t in winners]
            else:
                # Final Four and Championship - merge all
                all_remaining = []
                for teams in remaining.values():
                    all_remaining.extend(teams)
                remaining = {'All': [t for t in all_remaining if t in winners]}
        
        champion = winners[0].name if winners else None
        final_four = round_winners.get('Elite 8', [])
        
        return SimulationResult(
            champion=champion,
            final_four=final_four,
            round_winners=round_winners
        )
    
    def _simulate_round(self, remaining: dict, round: Round) -> List[TournamentTeam]:
        """Simulate all games in a round."""
        winners = []
        
        if round in [Round.FINAL_FOUR, Round.CHAMPIONSHIP]:
            # Cross-region matchups
            teams = remaining.get('All', [])
            for i in range(0, len(teams), 2):
                if i + 1 < len(teams):
                    winner = self.simulate_game(teams[i], teams[i+1], round)
                    winners.append(winner)
        else:
            # Within-region matchups
            for region in Region:
                teams = remaining.get(region, [])
                sorted_teams = sorted(teams, key=lambda t: t.seed)
                
                # Pair by bracket position
                n = len(sorted_teams)
                for i in range(n // 2):
                    team1 = sorted_teams[i]
                    team2 = sorted_teams[n - 1 - i]
                    winner = self.simulate_game(team1, team2, round)
                    winners.append(winner)
        
        return winners
    
    def _create_features(self, team1: TournamentTeam, team2: TournamentTeam,
                        round: Round) -> list:
        """Create feature vector for prediction."""
        return [
            team1.net_efficiency - team2.net_efficiency,
            team1.off_efficiency - team2.off_efficiency,
            team1.def_efficiency - team2.def_efficiency,
            team1.seed - team2.seed,
            team1.coach_ncaa_games - team2.coach_ncaa_games,
            team1.tempo - team2.tempo,
            list(Round).index(round) + 1  # Round number
        ]
    
    def run_simulations(self, bracket: Bracket, n_sims: int = 10000) -> dict:
        """Run multiple simulations and aggregate results."""
        
        championship_counts = {}
        final_four_counts = {}
        round_counts = {r.value: {} for r in Round}
        
        for _ in range(n_sims):
            result = self.simulate_tournament(bracket)
            
            # Count champion
            champ = result.champion
            championship_counts[champ] = championship_counts.get(champ, 0) + 1
            
            # Count Final Four
            for team in result.final_four:
                final_four_counts[team] = final_four_counts.get(team, 0) + 1
            
            # Count round winners
            for round_name, winners in result.round_winners.items():
                for team in winners:
                    round_counts[round_name][team] = round_counts[round_name].get(team, 0) + 1
        
        # Convert to probabilities
        return {
            'n_simulations': n_sims,
            'championship_probs': {k: v/n_sims for k, v in championship_counts.items()},
            'final_four_probs': {k: v/n_sims for k, v in final_four_counts.items()},
            'round_probs': {
                round_name: {k: v/n_sims for k, v in counts.items()}
                for round_name, counts in round_counts.items()
            },
            'most_likely_champion': max(championship_counts, key=championship_counts.get)
        }
```

## Phase 4: Visualization

### Task 4.1: Streamlit Bracket View

```python
# File: bracket_view.py

import streamlit as st
import pandas as pd
from pathlib import Path

def render_full_bracket():
    """Main bracket visualization page."""
    
    st.set_page_config(page_title="Bracket Oracle", layout="wide")
    
    st.title("🏀 March Madness Bracket Predictions")
    
    # Load data
    bracket_data = load_bracket_predictions()
    simulation_results = load_simulation_results()
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🏆 Full Bracket",
        "📊 Probabilities",
        "🚨 Upset Watch",
        "🎃 Cinderellas"
    ])
    
    with tab1:
        render_bracket_view(bracket_data)
    
    with tab2:
        render_probability_table(simulation_results)
    
    with tab3:
        render_upset_watch(bracket_data)
    
    with tab4:
        render_cinderella_view(simulation_results)

def render_bracket_view(data: dict):
    """Render the bracket visualization."""
    
    st.subheader("Tournament Bracket")
    
    # Champion
    champion = data.get('champion', 'TBD')
    st.success(f"🏆 Predicted Champion: **{champion}**")
    
    # Final Four
    st.subheader("Final Four")
    cols = st.columns(4)
    final_four = data.get('final_four', [])
    for i, team in enumerate(final_four):
        with cols[i]:
            st.metric(f"#{i+1}", team)
    
    # Regional brackets
    st.divider()
    for region in ['East', 'West', 'South', 'Midwest']:
        with st.expander(f"📍 {region} Region"):
            render_region(data, region)

def render_region(data: dict, region: str):
    """Render a single region bracket."""
    
    rounds = ['Round of 64', 'Round of 32', 'Sweet 16', 'Elite 8']
    cols = st.columns(len(rounds))
    
    for i, round_name in enumerate(rounds):
        with cols[i]:
            st.caption(round_name)
            games = [g for g in data.get('predictions', [])
                    if g.get('region') == region and g.get('round') == round_name]
            
            for game in games:
                prob = game.get('win_probability', 0.5)
                color = "🟢" if prob > 0.7 else "🟡" if prob > 0.55 else "🔴"
                upset = "⚠️" if game.get('is_upset') else ""
                st.markdown(f"{color} ({game['winner_seed']}) {game['winner']} {upset}")

def render_probability_table(results: dict):
    """Render probability heatmap/table."""
    
    st.subheader("Team Advancement Probabilities")
    
    # Create DataFrame
    rows = []
    for team, probs in results.get('team_probs', {}).items():
        rows.append({
            'Team': team,
            'Seed': probs['seed'],
            'Region': probs['region'],
            'R32': f"{probs['r32_prob']:.0%}",
            'S16': f"{probs['s16_prob']:.0%}",
            'E8': f"{probs['e8_prob']:.0%}",
            'FF': f"{probs['ff_prob']:.0%}",
            'Finals': f"{probs['finals_prob']:.0%}",
            'Champion': f"{probs['champ_prob']:.1%}"
        })
    
    df = pd.DataFrame(rows)
    
    # Sort options
    sort_by = st.selectbox("Sort by:", ['Champion', 'FF', 'Seed'], index=0)
    ascending = sort_by == 'Seed'
    
    st.dataframe(
        df.sort_values(sort_by, ascending=ascending),
        hide_index=True,
        width='stretch'
    )

def load_bracket_predictions() -> dict:
    """Load bracket predictions from file."""
    path = Path("data_files/tournament/predictions.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}

def load_simulation_results() -> dict:
    """Load simulation results from file."""
    path = Path("data_files/tournament/simulation_results.json")
    if path.exists():
        return json.loads(path.read_text())
    return {}
```

## Phase 5: Integration

### Task 5.1: Main Pipeline

```python
# File: run_bracket_predictions.py

"""
Main script to run complete bracket prediction pipeline.
"""

import argparse
from pathlib import Path
import json

from bracket_structures import Bracket
from fetch_bracket import fetch_current_bracket
from train_tournament_model import train_tournament_predictor
from bracket_simulator import BracketSimulator
import joblib

def main():
    parser = argparse.ArgumentParser(description="March Madness Bracket Predictor")
    parser.add_argument('--fetch', action='store_true', help='Fetch latest bracket')
    parser.add_argument('--train', action='store_true', help='Train models')
    parser.add_argument('--simulate', type=int, default=10000, help='Number of simulations')
    parser.add_argument('--output', type=str, default='data_files/tournament', help='Output directory')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Fetch bracket (if requested)
    if args.fetch:
        print("Fetching current bracket...")
        bracket_data = fetch_current_bracket()
        
        with open(output_dir / "bracket_2025.json", 'w') as f:
            json.dump(bracket_data, f, indent=2)
        print(f"Saved bracket to {output_dir / 'bracket_2025.json'}")
    
    # Step 2: Train models (if requested)
    if args.train:
        print("Training models...")
        train_tournament_predictor()
    
    # Step 3: Load models and bracket
    print("Loading models...")
    predictor = joblib.load("data_files/models/tournament_predictor.joblib")
    
    print("Loading bracket...")
    with open(output_dir / "bracket_2025.json") as f:
        bracket_data = json.load(f)
    
    bracket = build_bracket_from_data(bracket_data)
    
    # Step 4: Run simulations
    print(f"Running {args.simulate:,} simulations...")
    simulator = BracketSimulator(predictor)
    results = simulator.run_simulations(bracket, n_sims=args.simulate)
    
    # Step 5: Generate most likely bracket
    print("Generating most likely bracket...")
    most_likely = generate_most_likely_bracket(predictor, bracket)
    
    # Step 6: Save results
    with open(output_dir / "simulation_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(output_dir / "predictions.json", 'w') as f:
        json.dump(most_likely, f, indent=2)
    
    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("BRACKET PREDICTION SUMMARY")
    print("=" * 60)
    print(f"\n🏆 Most Likely Champion: {results['most_likely_champion']}")
    print(f"   Championship Probability: {results['championship_probs'].get(results['most_likely_champion'], 0):.1%}")
    
    print("\n📊 Top 5 Championship Contenders:")
    top_5 = sorted(results['championship_probs'].items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (team, prob) in enumerate(top_5, 1):
        print(f"   {i}. {team}: {prob:.1%}")
    
    print(f"\n✅ Results saved to {output_dir}")

def build_bracket_from_data(data: dict) -> Bracket:
    """Build Bracket object from JSON data."""
    # Implementation depends on data structure
    pass

def generate_most_likely_bracket(predictor, bracket: Bracket) -> dict:
    """Generate bracket picking favorites."""
    # Implementation in bracket_simulator.py
    pass

if __name__ == "__main__":
    main()
```

### Task 5.2: GitHub Actions Workflow

```yaml
# File: .github/workflows/tournament-predictions.yml

name: Tournament Predictions

on:
  schedule:
    # Run daily during tournament (March 15 - April 10)
    - cron: '0 8 15-31 3 *'
    - cron: '0 8 1-10 4 *'
  workflow_dispatch:
    inputs:
      simulations:
        description: 'Number of simulations'
        default: '10000'
        required: false

jobs:
  predict:
    runs-on: ubuntu-latest
    
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: pip install -r requirements-actions.txt
    
    - name: Fetch latest bracket
      run: python run_bracket_predictions.py --fetch
    
    - name: Run simulations
      run: |
        python run_bracket_predictions.py \
          --simulate ${{ github.event.inputs.simulations || 10000 }}
    
    - name: Commit results
      run: |
        git config --local user.email "action@github.com"
        git config --local user.name "GitHub Action"
        git add data_files/tournament/
        git commit -m "🏀 Update tournament predictions" || echo "No changes"
        git push
```

## File Structure

```
march-madness/
├── bracket_structures.py       # Data structures
├── tournament_data_collection.py
├── fetch_bracket.py           # Fetch official bracket
├── train_tournament_model.py   # Model training
├── bracket_simulator.py        # Monte Carlo simulation
├── bracket_view.py            # Streamlit visualization
├── run_bracket_predictions.py  # Main pipeline
│
├── data_files/
│   ├── tournament/
│   │   ├── bracket_2025.json
│   │   ├── predictions.json
│   │   ├── simulation_results.json
│   │   └── historical_tournament_games.csv
│   └── models/
│       ├── tournament_predictor.joblib
│       └── upset_predictor.joblib
│
└── docs/
    ├── roadmap-tournament-overview.md
    ├── roadmap-tournament-data.md
    ├── roadmap-tournament-models.md
    ├── roadmap-bracket-simulation.md
    ├── roadmap-bracket-visualization.md
    ├── roadmap-upset-detection.md
    └── roadmap-implementation.md
```

## Next Steps

1. Start with Phase 1: Build data structures
2. Collect historical tournament data
3. Train tournament-specific models
4. Build simulation engine
5. Create visualization
6. Test end-to-end before Selection Sunday
