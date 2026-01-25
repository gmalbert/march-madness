"""
March Madness Data Collection and API Testing

This module handles data collection from the College Basketball Data API
for March Madness betting predictions.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Optional
import cbbd
from cbbd.rest import ApiException

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed

# Configuration - check for various possible environment variable names
API_KEY = (os.environ.get("CBBD_API_KEY") or 
           os.environ.get("COLLEGE_BASKETBALL_API_KEY") or
           os.environ.get("COLLEGE_BASKETBALL_DATA_API_KEY"))

if not API_KEY:
    raise ValueError(
        "API key not found. Set one of these environment variables:\n"
        "  CBBD_API_KEY\n"
        "  COLLEGE_BASKETBALL_API_KEY\n"
        "  COLLEGE_BASKETBALL_DATA_API_KEY\n"
        "Get your API key from https://collegebasketballdata.com/"
    )

configuration = cbbd.Configuration(
    host="https://api.collegebasketballdata.com",
    access_token=API_KEY
)

# Cache directory
DATA_DIR = Path("data_files")
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_api_client():
    """Returns a configured API client."""
    return cbbd.ApiClient(configuration)


def cache_data(filename: str, data: List) -> None:
    """Save data to cache file."""
    filepath = CACHE_DIR / f"{filename}.json"
    with open(filepath, "w") as f:
        # Convert objects to dict if they have to_dict method
        if data and hasattr(data[0], 'to_dict'):
            json.dump([item.to_dict() for item in data], f, indent=2, default=str)
        else:
            json.dump(data, f, indent=2, default=str)
    print(f"Cached {len(data)} items to {filepath}")


def load_cached(filename: str) -> Optional[List]:
    """Load data from cache if exists, or from bundled historical data."""
    filepath = CACHE_DIR / f"{filename}.json"
    if filepath.exists():
        with open(filepath, "r") as f:
            return json.load(f)
    
    # Check bundled historical data
    bundled_path = DATA_DIR / "historical_data.json.gz"
    if bundled_path.exists():
        import gzip
        with gzip.open(bundled_path, "rt") as f:
            data = json.load(f)
        if filename in data:
            return data[filename]
    
    return None


def fetch_games(year: int, season_type: str = "regular") -> List:
    """Fetch games for a given year and season type."""
    cache_filename = f"games_{year}_{season_type}"

    # Check cache first
    cached = load_cached(cache_filename)
    if cached:
        print(f"Loaded {len(cached)} games from cache for {year} {season_type}")
        return cached

    with get_api_client() as api_client:
        games_api = cbbd.GamesApi(api_client)
        try:
            games = games_api.get_games(season=year, season_type=season_type)
            cache_data(cache_filename, games)
            print(f"Fetched {len(games)} games for {year} {season_type}")
            return games
        except ApiException as e:
            print(f"Error fetching games: {e}")
            return []


def fetch_tournament_games(year: int) -> List:
    """Fetch March Madness tournament games specifically."""
    return fetch_games(year, season_type="postseason")


def fetch_betting_lines(year: int, season_type: str = "regular") -> List:
    """Fetch betting lines including spreads, over/unders, moneylines.
    
    Uses date range filtering to get all games (regular season has 6000+ games,
    which exceeds the 3000 game API pagination limit).
    """
    cache_filename = f"lines_{year}_{season_type}"

    # Check cache first
    cached = load_cached(cache_filename)
    if cached:
        print(f"Loaded {len(cached)} betting lines from cache for {year} {season_type}")
        return cached

    with get_api_client() as api_client:
        lines_api = cbbd.LinesApi(api_client)
        try:
            from datetime import datetime
            
            if season_type == "postseason":
                # Tournament games: March Madness proper
                # Selection Sunday is typically second Sunday in March
                # Championship is first Monday in April
                start_date = datetime(year, 3, 14)
                end_date = datetime(year, 4, 10)
                
                lines = lines_api.get_lines(
                    season=year,
                    start_date_range=start_date,
                    end_date_range=end_date
                )
                
            else:
                # Regular season: ~6000 games, exceeds 3000 limit
                # Fetch in chunks: Nov-Dec, Jan-Feb, Early March
                print(f"Fetching regular season in chunks to bypass 3000-game limit...")
                
                # Season year maps to academic year (e.g., 2023 = 2022-23 season)
                chunks = [
                    (datetime(year - 1, 11, 1), datetime(year - 1, 12, 31)),  # Nov-Dec
                    (datetime(year, 1, 1), datetime(year, 2, 28)),            # Jan-Feb
                    (datetime(year, 3, 1), datetime(year, 3, 13)),            # Early March
                ]
                
                all_lines = []
                for start, end in chunks:
                    chunk = lines_api.get_lines(
                        season=year,
                        start_date_range=start,
                        end_date_range=end
                    )
                    all_lines.extend(chunk)
                    print(f"  {start.strftime('%b %Y')}-{end.strftime('%b %Y')}: {len(chunk)} games")
                
                lines = all_lines
            
            cache_data(cache_filename, lines)
            
            # Count how many have actual betting lines
            with_lines = [g for g in lines if g.lines and len(g.lines) > 0]
            print(f"Fetched {len(lines)} games for {year} {season_type}, {len(with_lines)} with betting lines")
            
            return lines
        except ApiException as e:
            print(f"Error fetching betting lines: {e}")
            return []


def fetch_team_stats(year: int) -> List:
    """Fetch team season statistics."""
    cache_filename = f"team_stats_{year}"

    cached = load_cached(cache_filename)
    if cached:
        print(f"Loaded {len(cached)} team stats from cache for {year}")
        return cached

    with get_api_client() as api_client:
        stats_api = cbbd.StatsApi(api_client)
        try:
            stats = stats_api.get_team_season_stats(season=year)
            
            # Convert objects to dictionaries if needed
            if stats and hasattr(stats[0], 'to_dict'):
                stats = [item.to_dict() for item in stats]
            
            cache_data(cache_filename, stats)
            print(f"Fetched {len(stats)} team stats for {year}")
            return stats
        except ApiException as e:
            print(f"Error fetching team stats: {e}")
            return []


def fetch_adjusted_efficiency(year: int) -> List:
    """Fetch adjusted efficiency ratings (KenPom-style metrics)."""
    cache_filename = f"efficiency_{year}"

    cached = load_cached(cache_filename)
    if cached:
        print(f"Loaded {len(cached)} efficiency ratings from cache for {year}")
        return cached

    with get_api_client() as api_client:
        ratings_api = cbbd.RatingsApi(api_client)
        try:
            # Try without year parameter first
            efficiency = ratings_api.get_adjusted_efficiency()
            
            # Convert objects to dictionaries if needed
            if efficiency and hasattr(efficiency[0], 'to_dict'):
                efficiency = [item.to_dict() for item in efficiency]
            
            cache_data(cache_filename, efficiency)
            print(f"Fetched {len(efficiency)} efficiency ratings")
            return efficiency
        except ApiException as e:
            print(f"Error fetching efficiency ratings: {e}")
            return []


def fetch_team_games_with_lines(team: str, years: int = 5) -> List[Dict]:
    """Fetch team's games with betting lines for historical ATS analysis."""
    import datetime
    
    games_with_lines = []
    current_year = datetime.datetime.now().year
    
    # Look at completed historical seasons (exclude current year)
    for year in range(current_year - years, current_year):
        # Fetch games for this season
        games = fetch_games(year, "regular")
        lines = fetch_betting_lines(year, "regular")
        
        # Create lookup for lines by game ID
        lines_lookup = {}
        for line_data in lines:
            if hasattr(line_data, 'to_dict'):
                line_dict = line_data.to_dict()
            else:
                line_dict = line_data
            
            game_id = line_dict.get('id') or line_dict.get('gameId')
            if game_id:
                lines_lookup[game_id] = line_dict
        
        # Filter games for this team and add line data
        for game in games:
            if hasattr(game, 'to_dict'):
                game_dict = game.to_dict()
            else:
                game_dict = game
            
            home_team = game_dict.get('homeTeam') or game_dict.get('home_team')
            away_team = game_dict.get('awayTeam') or game_dict.get('away_team')
            
            if team.lower() in (home_team or '').lower() or team.lower() in (away_team or '').lower():
                # Add line data if available
                game_id = game_dict.get('id') or game_dict.get('gameId')
                line_data = lines_lookup.get(game_id, {})
                
                if line_data.get('lines'):
                    # Get the first (most recent) line
                    line = line_data['lines'][0] if isinstance(line_data['lines'], list) else line_data['lines']
                    
                    # Determine if team is home or away and get appropriate spread
                    is_home = team.lower() in (home_team or '').lower()
                    
                    spread = line.get('spread', 0)
                    if spread is None:
                        spread = 0
                    
                    if not is_home:
                        spread = -spread  # Flip spread for away team
                    
                    # Get margin from final score (use correct field names)
                    margin = None
                    home_points = game_dict.get('homePoints')
                    away_points = game_dict.get('awayPoints')
                    
                    if home_points is not None and away_points is not None:
                        if is_home:
                            margin = home_points - away_points
                        else:
                            margin = away_points - home_points
                    
                    game_with_line = {
                        'season': year,
                        'team': team,
                        'opponent': away_team if is_home else home_team,
                        'is_home': is_home,
                        'spread': spread,
                        'margin': margin,
                        'covered': margin is not None and margin > -spread if spread != 0 else None
                    }
                    
                    games_with_lines.append(game_with_line)
    
    return games_with_lines


def test_api_connection() -> bool:
    """Test basic API connection by fetching current year teams."""
    try:
        with get_api_client() as api_client:
            teams_api = cbbd.TeamsApi(api_client)
            teams = teams_api.get_teams()
            print(f"✅ API connection successful! Found {len(teams)} teams.")
            return True
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False


def test_betting_lines() -> bool:
    """Test fetching betting lines."""
    try:
        # Try to get lines for postseason
        lines = fetch_betting_lines(2024, "postseason")
        if lines:
            print(f"✅ Betting lines fetch successful! Found {len(lines)} games with lines.")
            # Show sample
            if lines:
                sample = lines[0]
                print(f"Sample game: {sample['homeTeam']} vs {sample['awayTeam']}")
                if sample.get('lines') and len(sample['lines']) > 0:
                    line = sample['lines'][0]
                    print(f"  Spread: {line.get('spread', 'N/A')}")
                    print(f"  Over/Under: {line.get('overUnder', 'N/A')}")
                    print(f"  Home Moneyline: {line.get('homeMoneyline', 'N/A')}")
            return True
        else:
            print("⚠️  No betting lines found (may be expected for future tournaments)")
            return True
    except Exception as e:
        print(f"❌ Betting lines fetch failed: {e}")
        return False


def test_efficiency_ratings() -> bool:
    """Test fetching efficiency ratings."""
    try:
        efficiency = fetch_adjusted_efficiency(2024)
        if efficiency:
            print(f"✅ Efficiency ratings fetch successful! Found {len(efficiency)} teams.")
            # Show sample
            if efficiency:
                sample = efficiency[0]
                if isinstance(sample, dict):
                    print(f"Sample team: {sample.get('team', 'Unknown')}")
                    print(f"  Adj Offense: {sample.get('adjOffense', 'N/A')}")
                    print(f"  Adj Defense: {sample.get('adjDefense', 'N/A')}")
                else:
                    print(f"Sample team: {getattr(sample, 'team', 'Unknown')}")
                    print(f"  Adj Offense: {getattr(sample, 'adj_offense', 'N/A')}")
                    print(f"  Adj Defense: {getattr(sample, 'adj_defense', 'N/A')}")
            return True
        else:
            print("⚠️  No efficiency ratings found")
            return False
    except Exception as e:
        print(f"❌ Efficiency ratings fetch failed: {e}")
        return False


def collect_historical_betting_data(start_year: int, end_year: int):
    """Collect all data needed for betting model training."""
    print(f"Collecting data from {start_year} to {end_year}...")
    
    # Note: Betting lines API doesn't filter by year, so we get all available
    # We'll need to filter by date in post-processing
    
    # Get all available betting lines
    lines = fetch_betting_lines(2024, "all")  # Year parameter ignored
    print(f"Available betting lines: {len(lines)}")
    
    # Get efficiency ratings (current season)
    efficiency = fetch_adjusted_efficiency(2024)
    print(f"Efficiency ratings: {len(efficiency)}")
    
    # For historical games, we'll need to fetch by year
    for year in range(start_year, end_year + 1):
        print(f"Processing {year}...")
        
        # Regular season games
        reg_games = fetch_games(year, "regular")
        print(f"  Regular season games: {len(reg_games)}")
        
        # Tournament games
        tourn_games = fetch_tournament_games(year)
        print(f"  Tournament games: {len(tourn_games)}")
        
        # Team stats
        team_stats = fetch_team_stats(year)
        print(f"  Team stats: {len(team_stats)}")
    
    print("Data collection complete!")


def collect_historical_betting_data(start_year: int, end_year: int):
    """Collect all data needed for betting model training."""
    for year in range(start_year, end_year + 1):
        print(f"Collecting {year} data...")

        # Core data
        games = fetch_games(year)
        cache_data(f"games_{year}", games)

        tournament = fetch_tournament_games(year)
        cache_data(f"tournament_{year}", tournament)

        # Betting lines
        lines = fetch_betting_lines(year, "postseason")
        cache_data(f"lines_tournament_{year}", lines)

        # Team stats
        stats = fetch_team_stats(year)
        cache_data(f"team_stats_{year}", stats)

        # Ratings
        efficiency = fetch_adjusted_efficiency(year)
        cache_data(f"efficiency_{year}", efficiency)

        print(f"  {year} complete")


# Implementation of roadmap functions

def fetch_all_tournament_games(start_year: int = 2016, end_year: int = 2025) -> List[Dict]:
    """Fetch all March Madness tournament games across multiple years."""
    all_games = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching tournament games for {year}...")
        games = fetch_tournament_games(year)
        if games:
            # Add year to each game for tracking
            for game in games:
                if isinstance(game, dict):
                    game['season_year'] = year
                else:
                    # Convert to dict and add season_year
                    game_dict = game.to_dict() if hasattr(game, 'to_dict') else dict(game.__dict__)
                    game_dict['season_year'] = year
                    game = game_dict
            all_games.extend(games)
            print(f"  Found {len(games)} games for {year}")
        else:
            print(f"  No games found for {year}")
    
    print(f"Total tournament games collected: {len(all_games)}")
    return all_games


def fetch_historical_lines(start_year: int = 2016, end_year: int = 2025) -> List[Dict]:
    """Fetch historical betting lines for tournament games across multiple years."""
    all_lines = []
    for year in range(start_year, end_year + 1):
        print(f"Fetching betting lines for {year}...")
        lines = fetch_betting_lines(year, "postseason")
        if lines:
            # Add year to each line for tracking
            for line in lines:
                if isinstance(line, dict):
                    line['season_year'] = year
                else:
                    # Convert to dict and add season_year
                    line_dict = line.to_dict() if hasattr(line, 'to_dict') else dict(line.__dict__)
                    line_dict['season_year'] = year
                    line = line_dict
            all_lines.extend(lines)
            print(f"  Found {len(lines)} lines for {year}")
        else:
            print(f"  No lines found for {year}")
    
    print(f"Total betting lines collected: {len(all_lines)}")
    return all_lines


def fetch_team_season_data(year: int) -> List[Dict]:
    """Fetch comprehensive team stats for a season (alias for fetch_team_stats)."""
    return fetch_team_stats(year)


def fetch_efficiency_ratings(year: int) -> List[Dict]:
    """Fetch adjusted efficiency ratings for a season (alias for fetch_adjusted_efficiency)."""
    return fetch_adjusted_efficiency(year)


def fetch_rankings(year: int, week: int = None) -> List[Dict]:
    """Fetch poll rankings for a given year and optional week."""
    cache_filename = f"rankings_{year}" + (f"_week_{week}" if week else "_all")

    # Check cache first
    cached = load_cached(cache_filename)
    if cached:
        print(f"Loaded {len(cached)} rankings from cache for {year}" + (f" week {week}" if week else ""))
        return cached

    with get_api_client() as api_client:
        rankings_api = cbbd.RankingsApi(api_client)
        try:
            if week:
                rankings = rankings_api.get_rankings(season=year, week=week)
            else:
                rankings = rankings_api.get_rankings(season=year)
            cache_data(cache_filename, rankings)
            print(f"Fetched {len(rankings)} rankings for {year}" + (f" week {week}" if week else ""))
            return rankings
        except ApiException as e:
            print(f"Error fetching rankings: {e}")
            return []


def collect_comprehensive_betting_data(start_year: int = 2016, end_year: int = 2025):
    """Collect all data sets needed for comprehensive betting model training."""
    print(f"🔄 Collecting comprehensive betting data from {start_year} to {end_year}")
    print("=" * 60)

    # 1. Tournament games with results
    print("\n🏀 Collecting tournament games...")
    tournament_games = fetch_all_tournament_games(start_year, end_year)
    print(f"✅ Collected {len(tournament_games)} tournament games")

    # 2. Historical betting lines
    print("\n💰 Collecting betting lines...")
    betting_lines = fetch_historical_lines(start_year, end_year)
    print(f"✅ Collected {len(betting_lines)} betting lines")

    # 3. Team season statistics
    print("\n📊 Collecting team season statistics...")
    team_stats_data = {}
    for year in range(start_year, end_year + 1):
        stats = fetch_team_season_data(year)
        team_stats_data[year] = stats
        print(f"  {year}: {len(stats)} teams")
    total_team_stats = sum(len(stats) for stats in team_stats_data.values())
    print(f"✅ Collected {total_team_stats} team stat records")

    # 4. Efficiency ratings
    print("\n⚡ Collecting efficiency ratings...")
    efficiency_data = {}
    for year in range(start_year, end_year + 1):
        efficiency = fetch_efficiency_ratings(year)
        efficiency_data[year] = efficiency
        print(f"  {year}: {len(efficiency)} teams")
    total_efficiency = sum(len(eff) for eff in efficiency_data.values())
    print(f"✅ Collected {total_efficiency} efficiency ratings")

    # 5. Rankings (optional - get final rankings for each season)
    print("\n🏆 Collecting final rankings...")
    rankings_data = {}
    for year in range(start_year, end_year + 1):
        rankings = fetch_rankings(year)  # Get all rankings for the season
        rankings_data[year] = rankings
        print(f"  {year}: {len(rankings)} ranking entries")
    total_rankings = sum(len(rankings) for rankings in rankings_data.values())
    print(f"✅ Collected {total_rankings} ranking entries")

    print("\n" + "=" * 60)
    print("📋 Data Collection Summary:")
    print(f"  Tournament Games: {len(tournament_games)}")
    print(f"  Betting Lines: {len(betting_lines)}")
    print(f"  Team Statistics: {total_team_stats}")
    print(f"  Efficiency Ratings: {total_efficiency}")
    print(f"  Rankings: {total_rankings}")
    print("\n🎯 Ready for model training!")

    return {
        'tournament_games': tournament_games,
        'betting_lines': betting_lines,
        'team_stats': team_stats_data,
        'efficiency_ratings': efficiency_data,
        'rankings': rankings_data
    }


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--collect":
        print("🚀 Starting comprehensive data collection (2016-2025)...")
        collect_comprehensive_betting_data(2016, 2025)
    else:
        # Run API tests
        print("🔍 Testing CBBD API Connection...")
        print("=" * 50)

        # Test basic connection
        api_ok = test_api_connection()
        print()

        if api_ok:
            # Test betting lines
            lines_ok = test_betting_lines()
            print()

            # Test efficiency ratings
            efficiency_ok = test_efficiency_ratings()
            print()

            # Test new comprehensive data collection functions
            print("🧪 Testing comprehensive data collection functions...")
            print("-" * 50)

            # Test individual functions
            print("Testing fetch_all_tournament_games...")
            try:
                games_2024 = fetch_all_tournament_games(2024, 2024)
                print(f"✅ fetch_all_tournament_games: {len(games_2024)} games")
            except Exception as e:
                print(f"❌ fetch_all_tournament_games failed: {e}")

            print("Testing fetch_historical_lines...")
            try:
                lines_2024 = fetch_historical_lines(2024, 2024)
                print(f"✅ fetch_historical_lines: {len(lines_2024)} lines")
            except Exception as e:
                print(f"❌ fetch_historical_lines failed: {e}")

            print("Testing fetch_rankings...")
            try:
                rankings_2024 = fetch_rankings(2024)
                print(f"✅ fetch_rankings: {len(rankings_2024)} rankings")
            except Exception as e:
                print(f"❌ fetch_rankings failed: {e}")

            # Summary
            print("=" * 50)
            print("API Test Summary:")
            print(f"  Connection: {'✅' if api_ok else '❌'}")
            print(f"  Betting Lines: {'✅' if lines_ok else '❌'}")
            print(f"  Efficiency Ratings: {'✅' if efficiency_ok else '❌'}")

            if api_ok and lines_ok and efficiency_ok:
                print("\n🎉 All API tests passed! Ready to collect data.")
                print("\n🔄 To run comprehensive data collection (2016-2025), run:")
                print("    python data_collection.py --collect")
            else:
                print("\n⚠️  Some tests failed. Check your API key and network connection.")
        else:
            print("❌ Cannot proceed without API connection. Check your CBBD_API_KEY environment variable.")