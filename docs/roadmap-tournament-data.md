# Roadmap: Tournament Data Sources

*Data collection strategy for March Madness predictions.*

## Required Data Categories

### 1. Team Performance Data
- Season statistics (already have via CBBD API)
- Efficiency ratings (KenPom, BartTorvik)
- Four Factors (eFG%, TO%, ORB%, FTR)
- Strength of schedule

### 2. Tournament-Specific Data
- Official bracket and seedings
- Tournament history (team performance in March)
- Coach tournament experience
- Travel distance to game locations

### 3. Historical Tournament Data
- Past tournament results (2010-2024)
- Seed performance by round
- Upset frequency by matchup type

## Data Sources

### Primary Sources

```python
DATA_SOURCES = {
    'cbbd_api': {
        'url': 'https://api.collegebasketballdata.com',
        'data': ['team_stats', 'efficiency', 'rankings', 'games'],
        'access': 'API key required',
        'status': 'Active - already integrated'
    },
    'espn': {
        'url': 'https://site.api.espn.com/apis/site/v2/sports/basketball',
        'data': ['schedules', 'scores', 'brackets', 'standings'],
        'access': 'Free public API',
        'status': 'Active - already integrated'
    },
    'kenpom': {
        'url': 'https://kenpom.com',
        'data': ['efficiency_ratings', 'tempo', 'luck', 'sos'],
        'access': 'Subscription required ($20/year)',
        'status': 'Manual or scraping'
    },
    'barttorvik': {
        'url': 'https://barttorvik.com',
        'data': ['t-rank', 'projections', 'game_predictions'],
        'access': 'Free with rate limits',
        'status': 'Scraping possible'
    },
    'ncaa': {
        'url': 'https://www.ncaa.com/brackets',
        'data': ['official_bracket', 'seeds', 'regions'],
        'access': 'Free',
        'status': 'Scraping required'
    }
}
```

### Tournament Bracket Fetching

```python
import requests
from datetime import datetime

def fetch_tournament_bracket(year: int = None) -> dict:
    """Fetch official NCAA tournament bracket."""
    if year is None:
        year = datetime.now().year
    
    # ESPN Tournament API
    url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
    params = {
        'dates': f'{year}0301-{year}0410',  # March Madness window
        'groups': '100',  # NCAA Tournament
        'limit': 100
    }
    
    response = requests.get(url, params=params)
    data = response.json()
    
    return parse_bracket_data(data)

def parse_bracket_data(data: dict) -> dict:
    """Parse ESPN data into bracket structure."""
    bracket = {
        'year': data.get('season', {}).get('year'),
        'regions': {},
        'games': [],
        'teams': {}
    }
    
    for event in data.get('events', []):
        game_info = {
            'id': event['id'],
            'date': event['date'],
            'round': event.get('seasonType', {}).get('name'),
            'status': event.get('status', {}).get('type', {}).get('name')
        }
        
        for competition in event.get('competitions', []):
            for team in competition.get('competitors', []):
                team_data = {
                    'id': team['id'],
                    'name': team['team']['displayName'],
                    'seed': team.get('curatedRank', {}).get('current'),
                    'score': team.get('score'),
                    'winner': team.get('winner', False)
                }
                game_info[f"{'home' if team['homeAway'] == 'home' else 'away'}"] = team_data
        
        bracket['games'].append(game_info)
    
    return bracket
```

### Historical Tournament Data

```python
import pandas as pd

def load_historical_tournament_data():
    """Load historical tournament results for training."""
    # Kaggle NCAA Tournament dataset
    # https://www.kaggle.com/c/mens-march-mania-2024/data
    
    historical_games = pd.DataFrame()
    
    # Structure for historical data
    columns = [
        'year', 'round', 'region',
        'team1_id', 'team1_name', 'team1_seed', 'team1_score',
        'team2_id', 'team2_name', 'team2_seed', 'team2_score',
        'winner_id', 'upset'
    ]
    
    return historical_games

def calculate_historical_seed_performance():
    """Calculate how each seed performs historically."""
    seed_stats = {
        1: {'win_pct_r64': 0.99, 'win_pct_r32': 0.85, 'ff_pct': 0.40, 'champ_pct': 0.25},
        2: {'win_pct_r64': 0.94, 'win_pct_r32': 0.65, 'ff_pct': 0.20, 'champ_pct': 0.12},
        3: {'win_pct_r64': 0.85, 'win_pct_r32': 0.55, 'ff_pct': 0.12, 'champ_pct': 0.06},
        4: {'win_pct_r64': 0.80, 'win_pct_r32': 0.45, 'ff_pct': 0.08, 'champ_pct': 0.03},
        5: {'win_pct_r64': 0.65, 'win_pct_r32': 0.35, 'ff_pct': 0.05, 'champ_pct': 0.02},
        6: {'win_pct_r64': 0.63, 'win_pct_r32': 0.30, 'ff_pct': 0.03, 'champ_pct': 0.01},
        7: {'win_pct_r64': 0.60, 'win_pct_r32': 0.25, 'ff_pct': 0.02, 'champ_pct': 0.01},
        8: {'win_pct_r64': 0.50, 'win_pct_r32': 0.15, 'ff_pct': 0.02, 'champ_pct': 0.005},
        9: {'win_pct_r64': 0.50, 'win_pct_r32': 0.12, 'ff_pct': 0.01, 'champ_pct': 0.003},
        10: {'win_pct_r64': 0.40, 'win_pct_r32': 0.10, 'ff_pct': 0.01, 'champ_pct': 0.002},
        11: {'win_pct_r64': 0.37, 'win_pct_r32': 0.08, 'ff_pct': 0.02, 'champ_pct': 0.005},  # Play-in boost
        12: {'win_pct_r64': 0.35, 'win_pct_r32': 0.06, 'ff_pct': 0.005, 'champ_pct': 0.001},
        13: {'win_pct_r64': 0.20, 'win_pct_r32': 0.02, 'ff_pct': 0.001, 'champ_pct': 0.0},
        14: {'win_pct_r64': 0.15, 'win_pct_r32': 0.01, 'ff_pct': 0.0, 'champ_pct': 0.0},
        15: {'win_pct_r64': 0.06, 'win_pct_r32': 0.005, 'ff_pct': 0.0, 'champ_pct': 0.0},
        16: {'win_pct_r64': 0.01, 'win_pct_r32': 0.0, 'ff_pct': 0.0, 'champ_pct': 0.0},
    }
    return seed_stats
```

## Tournament-Specific Features

```python
def calculate_tournament_features(team_data: dict) -> dict:
    """Calculate features specific to tournament prediction."""
    
    features = {}
    
    # Experience features
    features['coach_tournament_games'] = team_data.get('coach_ncaa_games', 0)
    features['coach_final_fours'] = team_data.get('coach_final_fours', 0)
    features['program_tournament_apps'] = team_data.get('program_ncaa_apps', 0)
    features['years_since_last_tournament'] = team_data.get('years_since_ncaa', 0)
    
    # Seed-based features
    features['seed'] = team_data.get('seed', 16)
    features['seed_historical_ff_pct'] = get_seed_ff_probability(features['seed'])
    
    # Momentum features
    features['last_10_wins'] = team_data.get('last_10_record', [0, 0])[0]
    features['conference_tournament_result'] = team_data.get('conf_tourney_finish', 0)
    
    # Style features (important for matchups)
    features['tempo'] = team_data.get('tempo', 70)
    features['three_point_rate'] = team_data.get('three_rate', 0.35)
    features['turnover_rate'] = team_data.get('to_rate', 0.15)
    
    return features

def get_matchup_features(team1: dict, team2: dict) -> dict:
    """Calculate matchup-specific features."""
    
    features = {}
    
    # Seed differential
    features['seed_diff'] = team1['seed'] - team2['seed']
    
    # Efficiency differential
    features['net_eff_diff'] = team1['net_efficiency'] - team2['net_efficiency']
    features['off_eff_diff'] = team1['off_efficiency'] - team2['off_efficiency']
    features['def_eff_diff'] = team1['def_efficiency'] - team2['def_efficiency']
    
    # Style matchup
    features['tempo_diff'] = team1['tempo'] - team2['tempo']
    features['three_rate_diff'] = team1['three_rate'] - team2['three_rate']
    
    # Experience differential
    features['coach_exp_diff'] = team1['coach_tournament_games'] - team2['coach_tournament_games']
    
    return features
```

## Data Pipeline

```python
from dataclasses import dataclass
from typing import List, Dict
import json
from pathlib import Path

@dataclass
class TournamentTeam:
    """Tournament team with all relevant data."""
    id: str
    name: str
    seed: int
    region: str
    
    # Performance stats
    wins: int
    losses: int
    net_efficiency: float
    off_efficiency: float
    def_efficiency: float
    tempo: float
    
    # Rankings
    ap_rank: int = None
    kenpom_rank: int = None
    
    # Tournament history
    coach_ncaa_games: int = 0
    program_final_fours: int = 0

@dataclass
class TournamentBracket:
    """Complete tournament bracket structure."""
    year: int
    regions: Dict[str, List[TournamentTeam]]
    games: List[Dict]
    
    def get_first_round_matchups(self) -> List[tuple]:
        """Get all first round matchups."""
        matchups = []
        for region_name, teams in self.regions.items():
            # Sort by seed
            sorted_teams = sorted(teams, key=lambda t: t.seed)
            # Pair matchups: 1v16, 8v9, 5v12, 4v13, 6v11, 3v14, 7v10, 2v15
            pairings = [(0, 15), (7, 8), (4, 11), (3, 12), (5, 10), (2, 13), (6, 9), (1, 14)]
            for high, low in pairings:
                matchups.append((sorted_teams[high], sorted_teams[low], region_name))
        return matchups
    
    def save(self, filepath: str):
        """Save bracket to JSON."""
        data = {
            'year': self.year,
            'regions': {r: [vars(t) for t in teams] for r, teams in self.regions.items()},
            'games': self.games
        }
        Path(filepath).write_text(json.dumps(data, indent=2))
    
    @classmethod
    def load(cls, filepath: str) -> 'TournamentBracket':
        """Load bracket from JSON."""
        data = json.loads(Path(filepath).read_text())
        regions = {
            r: [TournamentTeam(**t) for t in teams]
            for r, teams in data['regions'].items()
        }
        return cls(year=data['year'], regions=regions, games=data['games'])
```

## Data Storage Structure

```
data_files/
├── tournament/
│   ├── bracket_2025.json           # Official bracket
│   ├── team_seeds_2025.csv         # Team seedings
│   ├── predictions_2025.json       # Our predictions
│   └── results_2025.json           # Actual results (updated live)
├── historical/
│   ├── tournament_games_2010_2024.csv
│   ├── seed_performance.csv
│   └── upset_analysis.csv
└── current_season/
    ├── team_stats_2025.json        # Already exists
    ├── efficiency_ratings_2025.json
    └── rankings_2025.json
```

## Next Steps

1. Integrate KenPom data (subscription or scraping)
2. Build historical tournament dataset
3. Create bracket fetching pipeline for Selection Sunday
4. See `roadmap-tournament-models.md` for prediction models
