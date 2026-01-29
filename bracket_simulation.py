"""
Bracket Simulation Engine

Monte Carlo simulation for predicting full tournament outcomes.
Implements the roadmap-bracket-simulation.md specification.
"""

import numpy as np
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from copy import deepcopy
import json
from pathlib import Path

@dataclass
class Team:
    """Tournament team."""
    id: str
    name: str
    seed: int
    region: str
    stats: Dict = field(default_factory=dict)

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

@dataclass
class BracketState:
    """Current state of the tournament bracket."""
    teams: Dict[str, Team]
    regions: Dict[str, List[Team]]
    results: Dict[int, List[Team]] = field(default_factory=dict)

    def get_remaining_teams(self, round_num: int) -> List[Team]:
        """Get teams still alive entering a round."""
        if round_num == 1:
            return list(self.teams.values())
        return self.results.get(round_num - 1, [])

    def get_matchups(self, round_num: int) -> List[Tuple[Team, Team]]:
        """Get matchups for a specific round."""
        remaining = self.get_remaining_teams(round_num)

        if round_num <= 4:  # Regional rounds (64, 32, 16, 8)
            matchups = []
            for region in ['East', 'West', 'Midwest', 'South']:
                region_teams = [t for t in remaining if t.region == region]
                if len(region_teams) >= 2:
                    # Sort by seed for proper bracket positioning
                    region_teams.sort(key=lambda x: x.seed)
                    # Pair teams: 1vs16, 8vs9, 4vs13, 5vs12, 2vs15, 7vs10, 3vs14, 6vs11
                    pairs = self._pair_region_teams(region_teams, round_num)
                    matchups.extend(pairs)
            return matchups
        else:  # Final Four and Championship
            if round_num == 5:  # Final Four
                # Pair winners from each region
                winners_by_region = {}
                for team in remaining:
                    winners_by_region[team.region] = team
                # Standard Final Four pairing
                return [
                    (winners_by_region.get('East'), winners_by_region.get('West')),
                    (winners_by_region.get('Midwest'), winners_by_region.get('South'))
                ]
            elif round_num == 6:  # Championship
                # Two Final Four winners play
                return [(remaining[0], remaining[1])] if len(remaining) >= 2 else []

        return []

    def _pair_region_teams(self, teams: List[Team], round_num: int) -> List[Tuple[Team, Team]]:
        """Pair teams within a region for the current round."""
        if len(teams) < 2:
            return []

        # For simplicity, pair highest remaining seed vs lowest remaining seed
        teams_sorted = sorted(teams, key=lambda x: x.seed)
        pairs = []

        for i in range(len(teams_sorted) // 2):
            team1 = teams_sorted[i]
            team2 = teams_sorted[-(i+1)]
            if team1 != team2:
                pairs.append((team1, team2))

        return pairs

    def advance_winner(self, round_num: int, winner: Team):
        """Advance a team to the next round."""
        if round_num not in self.results:
            self.results[round_num] = []
        self.results[round_num].append(winner)

class BracketSimulator:
    """Monte Carlo bracket simulation engine."""

    def __init__(self, game_predictor=None):
        """
        Initialize simulator.

        Args:
            game_predictor: Function that takes (team1, team2) and returns prob_team1_wins
        """
        self.game_predictor = game_predictor or self._default_predictor

    def _default_predictor(self, team1: Team, team2: Team) -> float:
        """Default predictor based on seed difference."""
        seed_diff = team2.seed - team1.seed  # Positive if team1 is favored
        # Simple logistic function
        prob = 1 / (1 + np.exp(-seed_diff * 0.5))
        return prob

    def simulate_bracket(self, bracket_state: BracketState, num_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulations of the tournament.

        Args:
            bracket_state: Initial bracket state
            num_simulations: Number of brackets to simulate

        Returns:
            Dictionary with team probabilities for each round
        """
        team_stats = {}

        # Initialize stats for all teams
        for team in bracket_state.teams.values():
            team_stats[team.id] = {
                'team': team,
                'round_64_prob': 1.0,  # Always 1.0 for Round of 64
                'round_32_prob': 0.0,
                'sweet_16_prob': 0.0,
                'elite_8_prob': 0.0,
                'final_four_prob': 0.0,
                'championship_prob': 0.0,
                'winner_prob': 0.0,
                'simulations': 0
            }

        for sim in range(num_simulations):
            # Create fresh bracket state for this simulation
            sim_bracket = deepcopy(bracket_state)

            # Count this simulation for all teams
            for team in bracket_state.teams.values():
                team_stats[team.id]['simulations'] += 1

            # Simulate each round
            for round_num in range(1, 7):  # Rounds 1-6
                matchups = sim_bracket.get_matchups(round_num)

                for team1, team2 in matchups:
                    if team1 and team2:
                        # Predict winner
                        prob_team1_wins = self.game_predictor(team1, team2)
                        winner = team1 if random.random() < prob_team1_wins else team2

                        # Advance winner
                        sim_bracket.advance_winner(round_num, winner)

                # Record stats after all matchups in this round are complete
                remaining_teams = sim_bracket.get_remaining_teams(round_num + 1)
                for team in remaining_teams:
                    round_key = self._round_name(round_num + 1)
                    if round_key == '32':
                        team_stats[team.id]['round_32_prob'] += 1
                    elif round_key == 'sweet_16':
                        team_stats[team.id]['sweet_16_prob'] += 1
                    elif round_key == 'elite_8':
                        team_stats[team.id]['elite_8_prob'] += 1
                    elif round_key == 'final_four':
                        team_stats[team.id]['final_four_prob'] += 1
                    elif round_key == 'championship':
                        team_stats[team.id]['championship_prob'] += 1
                    elif round_key == 'winner':
                        team_stats[team.id]['winner_prob'] += 1

        # Calculate probabilities
        for team_id, stats in team_stats.items():
            total_sims = num_simulations
            for round_key in ['round_32_prob', 'sweet_16_prob',
                            'elite_8_prob', 'final_four_prob', 'championship_prob', 'winner_prob']:
                if round_key in stats:
                    stats[round_key] = stats[round_key] / total_sims if total_sims > 0 else 0

        return team_stats

    def _round_name(self, round_num: int) -> str:
        """Convert round number to name."""
        round_names = {
            1: '64',      # Round of 64
            2: '32',      # Round of 32
            3: 'sweet_16', # Sweet 16
            4: 'elite_8',  # Elite 8
            5: 'final_four', # Final Four
            6: 'championship', # Championship
            7: 'winner'   # Winner
        }
        return round_names.get(round_num, '64')

def create_bracket_from_data(team_data: Dict, game_predictor=None) -> Tuple[BracketState, BracketSimulator]:
    """
    Create bracket state and simulator from team data.

    Args:
        team_data: Dictionary with team information
        game_predictor: Optional custom predictor function

    Returns:
        Tuple of (BracketState, BracketSimulator)
    """
    teams = {}
    regions = {'East': [], 'West': [], 'Midwest': [], 'South': []}

    for team_info in team_data.get('teams', []):
        team = Team(
            id=str(team_info.get('id', team_info.get('name', ''))),
            name=team_info.get('name', ''),
            seed=team_info.get('seed', 16),
            region=team_info.get('region', 'East'),
            stats=team_info.get('stats', {})
        )
        teams[team.id] = team
        if team.region in regions:
            regions[team.region].append(team)

    bracket_state = BracketState(teams=teams, regions=regions)
    simulator = BracketSimulator(game_predictor=game_predictor)

    return bracket_state, simulator

def load_real_tournament_bracket(year: int = 2025) -> Dict:
    """
    Load real tournament bracket data with team stats.

    Args:
        year: Tournament year

    Returns:
        Dictionary with tournament data
    """
    try:
        # Try to load from ESPN data
        from data_collection import fetch_tournament_games
        from data_tools.efficiency_loader import EfficiencyDataLoader

        # Load tournament games
        games = fetch_tournament_games(year)
        print(f"Loaded {len(games)} tournament games from {year}")

        # Load efficiency data
        efficiency_loader = EfficiencyDataLoader()
        kenpom_df, bart_df = efficiency_loader.load_data()

        # Extract unique teams from games
        teams = {}
        for game in games:
            for team_key in ['home_team', 'away_team']:
                if team_key in game:
                    team_info = game[team_key]
                    team_id = str(team_info.get('id', team_info.get('name', '')))
                    if team_id not in teams:
                        # Get efficiency data
                        team_stats = {}
                        if kenpom_df is not None:
                            kenpom_data = kenpom_df[kenpom_df['TeamName'].str.contains(team_info.get('name', ''), case=False, na=False)]
                            if not kenpom_data.empty:
                                team_stats.update({
                                    'net_efficiency': kenpom_data.iloc[0].get('NetRtg', 0),
                                    'off_efficiency': kenpom_data.iloc[0].get('ORtg', 0),
                                    'def_efficiency': kenpom_data.iloc[0].get('DRtg', 0),
                                    'tempo': kenpom_data.iloc[0].get('AdjT', 70),
                                })

                        teams[team_id] = {
                            'id': team_id,
                            'name': team_info.get('name', ''),
                            'seed': team_info.get('seed', 16),  # Will need to be set properly
                            'region': team_info.get('region', 'TBD'),  # Will need to be set properly
                            'stats': team_stats
                        }

        return {
            'year': year,
            'teams': list(teams.values()),
            'games': games
        }

    except Exception as e:
        print(f"Error loading real tournament data: {e}")
        print("Falling back to sample data...")
        return load_sample_tournament_bracket(year)

def load_sample_tournament_bracket(year: int = 2025) -> Dict:
    """
    Load sample tournament bracket data for testing.

    This is a placeholder with realistic team data for development.
    """
    # Sample teams with realistic data
    sample_teams = [
        # East Region
        {'id': '1', 'name': 'Duke', 'seed': 1, 'region': 'East', 'stats': {'net_efficiency': 25.0, 'tempo': 72.0, 'three_rate': 0.32}},
        {'id': '16', 'name': 'American', 'seed': 16, 'region': 'East', 'stats': {'net_efficiency': 5.0, 'tempo': 68.0, 'three_rate': 0.38}},
        {'id': '8', 'name': 'Florida', 'seed': 8, 'region': 'East', 'stats': {'net_efficiency': 15.0, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '9', 'name': 'Boise State', 'seed': 9, 'region': 'East', 'stats': {'net_efficiency': 12.0, 'tempo': 69.0, 'three_rate': 0.36}},
        {'id': '4', 'name': 'Arizona', 'seed': 4, 'region': 'East', 'stats': {'net_efficiency': 20.0, 'tempo': 71.0, 'three_rate': 0.33}},
        {'id': '13', 'name': 'Akron', 'seed': 13, 'region': 'East', 'stats': {'net_efficiency': 8.0, 'tempo': 67.0, 'three_rate': 0.39}},
        {'id': '5', 'name': 'BYU', 'seed': 5, 'region': 'East', 'stats': {'net_efficiency': 18.0, 'tempo': 70.5, 'three_rate': 0.34}},
        {'id': '12', 'name': 'Duquesne', 'seed': 12, 'region': 'East', 'stats': {'net_efficiency': 10.0, 'tempo': 68.5, 'three_rate': 0.37}},
        {'id': '2', 'name': 'Alabama', 'seed': 2, 'region': 'East', 'stats': {'net_efficiency': 23.0, 'tempo': 71.5, 'three_rate': 0.31}},
        {'id': '15', 'name': 'Robert Morris', 'seed': 15, 'region': 'East', 'stats': {'net_efficiency': 6.0, 'tempo': 67.5, 'three_rate': 0.40}},
        {'id': '7', 'name': 'Clemson', 'seed': 7, 'region': 'East', 'stats': {'net_efficiency': 16.0, 'tempo': 69.5, 'three_rate': 0.35}},
        {'id': '10', 'name': 'New Mexico', 'seed': 10, 'region': 'East', 'stats': {'net_efficiency': 11.0, 'tempo': 68.0, 'three_rate': 0.36}},
        {'id': '3', 'name': 'Baylor', 'seed': 3, 'region': 'East', 'stats': {'net_efficiency': 22.0, 'tempo': 71.0, 'three_rate': 0.32}},
        {'id': '14', 'name': 'Colgate', 'seed': 14, 'region': 'East', 'stats': {'net_efficiency': 7.0, 'tempo': 67.0, 'three_rate': 0.38}},
        {'id': '6', 'name': 'Dayton', 'seed': 6, 'region': 'East', 'stats': {'net_efficiency': 17.0, 'tempo': 70.0, 'three_rate': 0.34}},
        {'id': '11', 'name': 'Nevada', 'seed': 11, 'region': 'East', 'stats': {'net_efficiency': 9.0, 'tempo': 68.5, 'three_rate': 0.37}},

        # West Region
        {'id': '17', 'name': 'Houston', 'seed': 1, 'region': 'West', 'stats': {'net_efficiency': 24.0, 'tempo': 72.5, 'three_rate': 0.33}},
        {'id': '32', 'name': 'Longwood', 'seed': 16, 'region': 'West', 'stats': {'net_efficiency': 4.0, 'tempo': 66.0, 'three_rate': 0.41}},
        {'id': '24', 'name': 'Nebraska', 'seed': 8, 'region': 'West', 'stats': {'net_efficiency': 14.0, 'tempo': 69.0, 'three_rate': 0.35}},
        {'id': '25', 'name': 'Texas A&M', 'seed': 9, 'region': 'West', 'stats': {'net_efficiency': 13.0, 'tempo': 68.5, 'three_rate': 0.36}},
        {'id': '20', 'name': 'Texas', 'seed': 4, 'region': 'West', 'stats': {'net_efficiency': 19.0, 'tempo': 70.5, 'three_rate': 0.34}},
        {'id': '29', 'name': 'Colorado State', 'seed': 13, 'region': 'West', 'stats': {'net_efficiency': 7.5, 'tempo': 67.5, 'three_rate': 0.39}},
        {'id': '21', 'name': 'Xavier', 'seed': 5, 'region': 'West', 'stats': {'net_efficiency': 17.5, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '28', 'name': 'Missouri', 'seed': 12, 'region': 'West', 'stats': {'net_efficiency': 9.5, 'tempo': 68.0, 'three_rate': 0.37}},
        {'id': '18', 'name': 'Kentucky', 'seed': 2, 'region': 'West', 'stats': {'net_efficiency': 22.5, 'tempo': 71.5, 'three_rate': 0.32}},
        {'id': '31', 'name': 'Vermont', 'seed': 15, 'region': 'West', 'stats': {'net_efficiency': 5.5, 'tempo': 66.5, 'three_rate': 0.40}},
        {'id': '23', 'name': 'Indiana', 'seed': 7, 'region': 'West', 'stats': {'net_efficiency': 15.5, 'tempo': 69.5, 'three_rate': 0.36}},
        {'id': '26', 'name': 'Saint Mary\'s', 'seed': 10, 'region': 'West', 'stats': {'net_efficiency': 10.5, 'tempo': 67.5, 'three_rate': 0.37}},
        {'id': '19', 'name': 'Marquette', 'seed': 3, 'region': 'West', 'stats': {'net_efficiency': 21.0, 'tempo': 71.0, 'three_rate': 0.33}},
        {'id': '30', 'name': 'Drake', 'seed': 14, 'region': 'West', 'stats': {'net_efficiency': 6.5, 'tempo': 67.0, 'three_rate': 0.38}},
        {'id': '22', 'name': 'Michigan State', 'seed': 6, 'region': 'West', 'stats': {'net_efficiency': 16.5, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '27', 'name': 'UC San Diego', 'seed': 11, 'region': 'West', 'stats': {'net_efficiency': 8.5, 'tempo': 68.0, 'three_rate': 0.38}},

        # Midwest Region
        {'id': '33', 'name': 'Kansas', 'seed': 1, 'region': 'Midwest', 'stats': {'net_efficiency': 23.5, 'tempo': 72.0, 'three_rate': 0.33}},
        {'id': '48', 'name': 'Southeast Missouri State', 'seed': 16, 'region': 'Midwest', 'stats': {'net_efficiency': 3.5, 'tempo': 65.5, 'three_rate': 0.42}},
        {'id': '40', 'name': 'UCLA', 'seed': 8, 'region': 'Midwest', 'stats': {'net_efficiency': 13.5, 'tempo': 69.0, 'three_rate': 0.36}},
        {'id': '41', 'name': 'Utah State', 'seed': 9, 'region': 'Midwest', 'stats': {'net_efficiency': 12.5, 'tempo': 68.5, 'three_rate': 0.37}},
        {'id': '36', 'name': 'Iowa State', 'seed': 4, 'region': 'Midwest', 'stats': {'net_efficiency': 18.5, 'tempo': 70.5, 'three_rate': 0.34}},
        {'id': '45', 'name': 'Lipscomb', 'seed': 13, 'region': 'Midwest', 'stats': {'net_efficiency': 7.0, 'tempo': 67.0, 'three_rate': 0.39}},
        {'id': '37', 'name': 'Maryland', 'seed': 5, 'region': 'Midwest', 'stats': {'net_efficiency': 17.0, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '44', 'name': 'Grand Canyon', 'seed': 12, 'region': 'Midwest', 'stats': {'net_efficiency': 8.0, 'tempo': 67.5, 'three_rate': 0.38}},
        {'id': '34', 'name': 'Tennessee', 'seed': 2, 'region': 'Midwest', 'stats': {'net_efficiency': 22.0, 'tempo': 71.5, 'three_rate': 0.32}},
        {'id': '47', 'name': 'Wagner', 'seed': 15, 'region': 'Midwest', 'stats': {'net_efficiency': 4.5, 'tempo': 66.0, 'three_rate': 0.41}},
        {'id': '39', 'name': 'Washington State', 'seed': 7, 'region': 'Midwest', 'stats': {'net_efficiency': 15.0, 'tempo': 69.5, 'three_rate': 0.36}},
        {'id': '42', 'name': 'North Carolina State', 'seed': 10, 'region': 'Midwest', 'stats': {'net_efficiency': 11.0, 'tempo': 68.0, 'three_rate': 0.37}},
        {'id': '35', 'name': 'Arkansas', 'seed': 3, 'region': 'Midwest', 'stats': {'net_efficiency': 20.5, 'tempo': 71.0, 'three_rate': 0.33}},
        {'id': '46', 'name': 'Norfolk State', 'seed': 14, 'region': 'Midwest', 'stats': {'net_efficiency': 6.0, 'tempo': 66.5, 'three_rate': 0.40}},
        {'id': '38', 'name': 'Memphis', 'seed': 6, 'region': 'Midwest', 'stats': {'net_efficiency': 16.0, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '43', 'name': 'Oklahoma', 'seed': 11, 'region': 'Midwest', 'stats': {'net_efficiency': 9.0, 'tempo': 68.5, 'three_rate': 0.38}},

        # South Region
        {'id': '49', 'name': 'Auburn', 'seed': 1, 'region': 'South', 'stats': {'net_efficiency': 23.0, 'tempo': 72.0, 'three_rate': 0.33}},
        {'id': '64', 'name': 'Alabama State', 'seed': 16, 'region': 'South', 'stats': {'net_efficiency': 3.0, 'tempo': 65.0, 'three_rate': 0.43}},
        {'id': '56', 'name': 'Louisville', 'seed': 8, 'region': 'South', 'stats': {'net_efficiency': 13.0, 'tempo': 69.0, 'three_rate': 0.36}},
        {'id': '57', 'name': 'Creighton', 'seed': 9, 'region': 'South', 'stats': {'net_efficiency': 12.0, 'tempo': 68.5, 'three_rate': 0.37}},
        {'id': '52', 'name': 'Wisconsin', 'seed': 4, 'region': 'South', 'stats': {'net_efficiency': 18.0, 'tempo': 70.5, 'three_rate': 0.34}},
        {'id': '61', 'name': 'James Madison', 'seed': 13, 'region': 'South', 'stats': {'net_efficiency': 6.5, 'tempo': 67.0, 'three_rate': 0.39}},
        {'id': '53', 'name': 'Florida Atlantic', 'seed': 5, 'region': 'South', 'stats': {'net_efficiency': 16.5, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '60', 'name': 'Vermont', 'seed': 12, 'region': 'South', 'stats': {'net_efficiency': 7.5, 'tempo': 67.5, 'three_rate': 0.38}},
        {'id': '50', 'name': 'Texas A&M', 'seed': 2, 'region': 'South', 'stats': {'net_efficiency': 21.5, 'tempo': 71.5, 'three_rate': 0.32}},
        {'id': '63', 'name': 'FDU', 'seed': 15, 'region': 'South', 'stats': {'net_efficiency': 4.0, 'tempo': 66.0, 'three_rate': 0.41}},
        {'id': '55', 'name': 'Michigan', 'seed': 7, 'region': 'South', 'stats': {'net_efficiency': 14.5, 'tempo': 69.5, 'three_rate': 0.36}},
        {'id': '58', 'name': 'Providence', 'seed': 10, 'region': 'South', 'stats': {'net_efficiency': 10.5, 'tempo': 68.0, 'three_rate': 0.37}},
        {'id': '51', 'name': 'Iowa', 'seed': 3, 'region': 'South', 'stats': {'net_efficiency': 20.0, 'tempo': 71.0, 'three_rate': 0.33}},
        {'id': '62', 'name': 'UAB', 'seed': 14, 'region': 'South', 'stats': {'net_efficiency': 5.5, 'tempo': 66.5, 'three_rate': 0.40}},
        {'id': '54', 'name': 'Northwestern', 'seed': 6, 'region': 'South', 'stats': {'net_efficiency': 15.5, 'tempo': 70.0, 'three_rate': 0.35}},
        {'id': '59', 'name': 'Richmond', 'seed': 11, 'region': 'South', 'stats': {'net_efficiency': 8.5, 'tempo': 68.5, 'three_rate': 0.38}},
    ]

    return {'year': year, 'teams': sample_teams}

def create_predictor_from_models(models: Dict = None, efficiency_data: Dict = None) -> callable:
    """
    Create a game predictor function using trained ML models.

    Args:
        models: Dictionary of trained models (spread, total, moneyline)
        efficiency_data: Team efficiency data

    Returns:
        Function that predicts P(team1 beats team2)
    """
    def predictor(team1: Team, team2: Team) -> float:
        """Predict probability that team1 beats team2."""
        try:
            # Use efficiency difference as primary predictor
            eff1 = team1.stats.get('net_efficiency', 10)
            eff2 = team2.stats.get('net_efficiency', 10)
            eff_diff = eff1 - eff2

            # Convert efficiency difference to probability
            # Each efficiency point is worth about 2% in win probability
            prob = 0.5 + (eff_diff * 0.02)

            # Bound between 0.05 and 0.95
            prob = max(0.05, min(0.95, prob))

            return prob

        except Exception:
            # Fallback to seed-based prediction
            seed_diff = team2.seed - team1.seed
            prob = 1 / (1 + np.exp(-seed_diff * 0.5))
            return prob

    return predictor