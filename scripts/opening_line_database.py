import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class OpeningLineDatabase:
    """
    A simple database to store and retrieve opening betting lines for college basketball games.
    This serves as a fallback when live APIs don't provide historical opening line data.
    """

    def __init__(self, data_dir: str = "data_files"):
        self.data_dir = Path(data_dir)
        self.opening_lines_file = self.data_dir / "opening_lines.json"
        self.data_dir.mkdir(exist_ok=True)

        # Initialize empty database if it doesn't exist
        if not self.opening_lines_file.exists():
            self._save_data({})

    def _load_data(self) -> Dict:
        """Load the opening lines data from file."""
        try:
            with open(self.opening_lines_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save_data(self, data: Dict):
        """Save the opening lines data to file."""
        with open(self.opening_lines_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_opening_line(self, game_id: str, home_team: str, away_team: str,
                        opening_spread: float, opening_total: float,
                        game_date: str, season: str = None):
        """
        Add an opening line for a game.

        Args:
            game_id: Unique identifier for the game
            home_team: Home team name
            away_team: Away team name
            opening_spread: Opening point spread (negative for favorite)
            opening_total: Opening over/under total
            game_date: Date of the game (YYYY-MM-DD format)
            season: Season identifier (e.g., "2023-24")
        """
        data = self._load_data()

        game_key = f"{season}_{game_id}" if season else game_id

        data[game_key] = {
            "home_team": home_team,
            "away_team": away_team,
            "opening_spread": opening_spread,
            "opening_total": opening_total,
            "game_date": game_date,
            "season": season,
            "added_timestamp": datetime.now().isoformat()
        }

        self._save_data(data)
        print(f"Added opening line for game: {home_team} vs {away_team}")

    def get_opening_line(self, game_id: str, season: str = None) -> Optional[Dict]:
        """
        Retrieve opening line for a specific game.

        Args:
            game_id: Unique identifier for the game
            season: Season identifier

        Returns:
            Dictionary with opening line data or None if not found
        """
        data = self._load_data()
        game_key = f"{season}_{game_id}" if season else game_id

        return data.get(game_key)

    def find_similar_games(self, team1: str, team2: str, limit: int = 5) -> List[Dict]:
        """
        Find games between similar teams to estimate opening lines.

        Args:
            team1: Name of first team
            team2: Name of second team
            limit: Maximum number of similar games to return

        Returns:
            List of similar games with their opening lines
        """
        data = self._load_data()
        similar_games = []

        # Simple similarity based on team names (could be improved with team IDs)
        for game_key, game_data in data.items():
            if ((team1.lower() in game_data['home_team'].lower() and
                 team2.lower() in game_data['away_team'].lower()) or
                (team1.lower() in game_data['away_team'].lower() and
                 team2.lower() in game_data['home_team'].lower())):
                similar_games.append(game_data)

        return similar_games[:limit]

    def get_season_stats(self, season: str) -> Dict:
        """
        Get statistics for a specific season.

        Args:
            season: Season identifier

        Returns:
            Dictionary with season statistics
        """
        data = self._load_data()
        season_games = {k: v for k, v in data.items() if v.get('season') == season}

        if not season_games:
            return {"total_games": 0, "avg_spread": 0, "avg_total": 0}

        spreads = [game['opening_spread'] for game in season_games.values()]
        totals = [game['opening_total'] for game in season_games.values()]

        return {
            "total_games": len(season_games),
            "avg_spread": sum(spreads) / len(spreads) if spreads else 0,
            "avg_total": sum(totals) / len(totals) if totals else 0,
            "spread_range": (min(spreads), max(spreads)) if spreads else (0, 0),
            "total_range": (min(totals), max(totals)) if totals else (0, 0)
        }

    def export_to_csv(self, filename: str):
        """
        Export the database to CSV format for analysis.

        Args:
            filename: Output CSV filename
        """
        import csv

        data = self._load_data()

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['game_id', 'season', 'home_team', 'away_team',
                         'opening_spread', 'opening_total', 'game_date', 'added_timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for game_key, game_data in data.items():
                row = {
                    'game_id': game_key,
                    'season': game_data.get('season', ''),
                    'home_team': game_data.get('home_team', ''),
                    'away_team': game_data.get('away_team', ''),
                    'opening_spread': game_data.get('opening_spread', 0),
                    'opening_total': game_data.get('opening_total', 0),
                    'game_date': game_data.get('game_date', ''),
                    'added_timestamp': game_data.get('added_timestamp', '')
                }
                writer.writerow(row)

        print(f"Exported {len(data)} games to {filename}")

# Example usage and sample data population
if __name__ == "__main__":
    db = OpeningLineDatabase()

    # Add some sample historical opening lines
    # These would typically come from SportsDataIO or other sources
    sample_games = [
        {
            "game_id": "2024_01_001",
            "home_team": "Duke",
            "away_team": "North Carolina",
            "opening_spread": -3.5,
            "opening_total": 145.5,
            "game_date": "2024-01-15",
            "season": "2023-24"
        },
        {
            "game_id": "2024_01_002",
            "home_team": "Kansas",
            "away_team": "Texas",
            "opening_spread": -2.0,
            "opening_total": 138.0,
            "game_date": "2024-01-20",
            "season": "2023-24"
        },
        {
            "game_id": "2024_02_001",
            "home_team": "UConn",
            "away_team": "Creighton",
            "opening_spread": -4.0,
            "opening_total": 142.5,
            "game_date": "2024-02-10",
            "season": "2023-24"
        }
    ]

    for game in sample_games:
        db.add_opening_line(**game)

    # Export to CSV for analysis
    db.export_to_csv("opening_lines_sample.csv")

    # Print season stats
    stats = db.get_season_stats("2023-24")
    print(f"Season 2023-24 stats: {stats}")