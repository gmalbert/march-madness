#!/usr/bin/env python3
"""
Fetch live betting odds from The Odds API for NCAA basketball games.

This module fetches current moneyline, spread, and total odds from multiple
sportsbooks and caches them for use in predictions and UI.
"""

import os
import json
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
DATA_DIR = Path("data_files")
CACHE_DIR = DATA_DIR / "cache"
CACHE_FILE = CACHE_DIR / "odds_live.json"
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4/sports/basketball_ncaab/odds"

def normalize_team_name(name: str) -> str:
    """Normalize team names for matching - matches predictions.py logic."""
    # Common patterns: remove mascots/nicknames
    # Split on space and take first part(s) that aren't mascots
    mascots = [
        'Wolverines', 'Hoosiers', 'Cyclones', 'Knights', 'Gators', 'Tigers',
        'Wolfpack', 'Dukes', 'Billikens', 'Bonnies', 'Buckeyes', 'Demon Deacons', 'Flashes', 'RedHawks', 'Ducks',
        'Spartans', 'Bears', 'Raiders', 'Razorbacks', 'Commodores', 'Bulldogs',
        'Bruins', 'Boilermakers', 'Buffaloes', 'Jayhawks', 'Wildcats', 'Aggies',
        'Huskies', 'Tar Heels', 'Blue Devils', 'Cardinals', 'Sooners', 'Longhorns',
        'Crimson Tide', 'Volunteers', 'Gamecocks', 'Rebels', 'Broncos', 'Cougars',
        'Panthers', 'Eagles', 'Owls', 'Rams', 'Bulls', 'Golden Knights', 'Mean Green',
        'Thundering Herd', 'Miners', 'Roadrunners', 'Hilltoppers', 'Golden Flashes',
        'Bearcats', 'Fighting Illini', 'Terrapins', 'Cornhuskers', 'Waves', 'Golden Gophers',
        'Cavaliers', 'Mountaineers', 'Hokies', 'Cowboys', 'Utes', 'Dons', 'Dolphins', 'Red Flash',
        'Chargers', 'Skyhawks', 'Lakers', 'Mastodons', 'Jaguars', 'Seahawks', 'Sharks', 'Salukis',
        'Purple Aces', 'Trojans', 'Badgers', 'Scarlet Knights', 'Friars', 'Revolutionaries', 'Minutemen', 'Horned Frogs', 'Flyers',
        'Braves', 'Cardinal', 'Flames', 'Gaels', 'Grizzlies', 'Jaspers', 'Kangaroos', 'Leopards', 'Mavericks', 'Pilots', 'Redhawks', 'Stags'
    ]

    # Multi-word mascots that need special handling
    multi_word_mascots = [
        'Tar Heels', 'Blue Devils', 'Fighting Irish', 'Golden Flashes', 'Red Raiders',
        'Golden Knights', 'Thundering Herd', 'Crimson Tide', 'Mean Green', 'Fighting Illini',
        'Demon Deacons', 'Golden Gophers', 'Yellow Jackets', 'Red Flash', 'Purple Aces', 'Scarlet Knights',
        'A&M Rattlers', 'Arizona Lumberjacks', 'Baptist Lancers', 'Beach St 49ers', 'Golden Lions',
        'Canyon Antelopes', 'Diego Toreros', 'Fullerton Titans', 'Golden Griffins', 'Mary\'s Gaels',
        'Michigan Chippewas', 'Mountain Hawks', 'Northridge Matadors', 'Rainbow Warriors',
        'San Diego Tritons', 'Santa Barbara Gauchos', 'St Bobcats', 'St Braves', 'St Sun Devils',
        'State Bengals', 'Tech Trailblazers', 'Utah Thunderbirds', 'Wolf Pack', 'Irvine Anteaters',
        'Mexico Lobos', 'Poly Mustangs'
    ]

    # Handle special cases BEFORE mascot removal
    special_cases = {
        # Existing mappings
        'Miami (FL)': 'Miami',
        'Miami (OH)': 'Miami (OH)',
        'NC State': 'North Carolina State',
        'Kent State Golden Flashes': 'Kent State',
        'Texas Tech Red Raiders': 'Texas Tech',
        'UCF': 'UCF',
        'UCLA': 'UCLA',
        'USC': 'USC',
        'LSU': 'LSU',
        'TCU': 'TCU',
        'SMU': 'SMU',
        'BYU': 'BYU',
        'VCU': 'VCU',
        'UNLV': 'UNLV',
        'IU Indianapolis Jaguars': 'IU Indianapolis',
        'IU Indianapolis': 'IU Indianapolis',
        'IUPUI': 'IU Indianapolis',
        'Long Island University Sharks': 'Long Island University',
        'LIU': 'Long Island University',
        'LIU Sharks': 'Long Island University',
        'Purdue Fort Wayne Mastodons': 'Purdue Fort Wayne',
        'Central Connecticut Blue Devils': 'Central Connecticut',
        'Chicago State Cougars': 'Chicago State',
        'Kansas St': 'Kansas State',
        
        # New mappings for Odds API abbreviations
        'Pacific': 'Pacific',
        'Seattle': 'Seattle',
        'Long': 'Long Beach State',
        'UC': 'UC Irvine',  # This might need refinement based on context
        'Cal': 'California',
        'CSU': 'Colorado State',
        'Fresno St': 'Fresno State',
        'Grand': 'Grand Canyon',
        'Utah Valley': 'Utah Valley',
        'UIC': 'Illinois Chicago',
        'Northern': 'Northern Arizona',
        'N Colorado': 'North Colorado',
        'Weber State': 'Weber State',
        'Saint': 'Saint Mary\'s',
        'New': 'New Mexico',
        'UMKC': 'UMKC',
        'Omaha': 'Omaha',
        'California Golden': 'California',
        'Southern': 'Southern Utah',
        'San': 'San Diego',
        'Santa Clara': 'Santa Clara',
        'Hawai\'i': 'Hawaii',
        'Stonehill': 'Stonehill',
        'Central Connecticut St': 'Central Connecticut',
        'Mercyhurst': 'Mercyhurst',
        'Chicago St': 'Chicago State',
        'Fort Wayne': 'Purdue Fort Wayne',
        
        # Additional common abbreviations
        'St': 'State',  # General St -> State mapping
        'N': 'North',   # N Colorado -> North Colorado
    }

    if name in special_cases:
        return special_cases[name]

    # Check for multi-word mascots first
    for mascot in multi_word_mascots:
        if name.endswith(' ' + mascot):
            name = name[:-len(' ' + mascot)].strip()
            # Check special cases again after mascot removal
            if name in special_cases:
                return special_cases[name]
            return name

    # Remove single-word mascot from end
    parts = name.split()
    if len(parts) > 1 and parts[-1] in mascots:
        name = ' '.join(parts[:-1])
        # Check special cases again after mascot removal
        if name in special_cases:
            return special_cases[name]
        return name

    return name


def fetch_live_odds(force_refresh: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Fetch live betting odds from The Odds API.

    Returns a dict keyed by normalized 'home_team vs away_team' with odds data.
    Caches results for 24 hours unless force_refresh=True.

    Args:
        force_refresh: If True, ignore cache and fetch fresh data

    Returns:
        Dict of game keys to odds data
    """
    if not API_KEY:
        raise ValueError("ODDS_API_KEY not found in environment. Please set it in .env file.")

    # Check cache first
    if not force_refresh and CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, 'r') as f:
                cached_data = json.load(f)

            # Check if cache is less than 24 hours old
            cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
            if datetime.now() - cache_time < timedelta(hours=24):
                print(f"Using cached odds from {cache_time}")
                return cached_data.get('odds', {})
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Cache file corrupted, fetching fresh data: {e}")

    # Fetch fresh data
    print("Fetching live odds from The Odds API...")
    
    # Fetch all upcoming games with active markets (no date filtering)
    # This provides the broadest coverage of games with available odds
    params = {
        'apiKey': API_KEY,
        'regions': 'us',  # US sportsbooks
        'markets': 'h2h,spreads,totals',  # Moneyline, spreads, totals
        'oddsFormat': 'american'  # American odds format
    }

    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        # Process and normalize the data
        odds_data = {}
        for game in data:
            home_team = normalize_team_name(game.get('home_team', ''))
            away_team = normalize_team_name(game.get('away_team', ''))

            if not home_team or not away_team:
                continue

            # Create game key
            game_key = f"{home_team} vs {away_team}"

            # Extract odds from bookmakers (check all for best coverage)
            bookmakers = game.get('bookmakers', [])
            if not bookmakers:
                continue

            game_odds = {}

            # Extract moneyline (h2h) - check all bookmakers
            for bookmaker in bookmakers:
                markets = bookmaker.get('markets', [])
                h2h_market = next((m for m in markets if m['key'] == 'h2h'), None)
                if h2h_market and 'home_moneyline' not in game_odds:
                    outcomes = h2h_market['outcomes']
                    home_outcome = next((o for o in outcomes if o['name'] == game['home_team']), None)
                    away_outcome = next((o for o in outcomes if o['name'] == game['away_team']), None)

                    if home_outcome and away_outcome:
                        game_odds['home_moneyline'] = home_outcome['price']
                        game_odds['away_moneyline'] = away_outcome['price']
                        game_odds['moneyline_bookmaker'] = bookmaker.get('title')
                        break  # Found moneyline, stop checking other bookmakers

            # Extract spread - check all bookmakers
            for bookmaker in bookmakers:
                markets = bookmaker.get('markets', [])
                spreads_market = next((m for m in markets if m['key'] == 'spreads'), None)
                if spreads_market and 'home_spread' not in game_odds:
                    outcomes = spreads_market['outcomes']
                    home_outcome = next((o for o in outcomes if o['name'] == game['home_team']), None)
                    away_outcome = next((o for o in outcomes if o['name'] == game['away_team']), None)

                    if home_outcome and away_outcome:
                        game_odds['home_spread'] = home_outcome['point']
                        game_odds['away_spread'] = away_outcome['point']
                        game_odds['home_spread_odds'] = home_outcome['price']
                        game_odds['away_spread_odds'] = away_outcome['price']
                        game_odds['spread_bookmaker'] = bookmaker.get('title')
                        break  # Found spread, stop checking other bookmakers

            # Extract total - check all bookmakers
            for bookmaker in bookmakers:
                markets = bookmaker.get('markets', [])
                totals_market = next((m for m in markets if m['key'] == 'totals'), None)
                if totals_market and 'total_line' not in game_odds:
                    outcomes = totals_market['outcomes']
                    over_outcome = next((o for o in outcomes if o['name'] == 'Over'), None)
                    under_outcome = next((o for o in outcomes if o['name'] == 'Under'), None)

                    if over_outcome and under_outcome:
                        game_odds['total_line'] = over_outcome['point']
                        game_odds['over_odds'] = over_outcome['price']
                        game_odds['under_odds'] = under_outcome['price']
                        game_odds['total_bookmaker'] = bookmaker.get('title')
                        break  # Found total, stop checking other bookmakers

            # Add game metadata
            game_odds['commence_time'] = game.get('commence_time')
            game_odds['bookmaker'] = game_odds.get('moneyline_bookmaker') or game_odds.get('spread_bookmaker') or game_odds.get('total_bookmaker') or 'Unknown'

            if game_odds:
                odds_data[game_key] = game_odds

        # Cache the results
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            'timestamp': datetime.now().isoformat(),
            'odds': odds_data
        }

        with open(CACHE_FILE, 'w') as f:
            json.dump(cache_payload, f, indent=2)

        print(f"Cached {len(odds_data)} games with live odds")
        return odds_data

    except requests.RequestException as e:
        print(f"Error fetching odds: {e}")
        # Return cached data if available, even if stale
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, 'r') as f:
                    cached_data = json.load(f)
                print("Returning stale cached odds due to API error")
                return cached_data.get('odds', {})
            except json.JSONDecodeError:
                pass
        return {}

if __name__ == "__main__":
    # Test the fetcher
    odds = fetch_live_odds(force_refresh=True)
    print(f"Fetched odds for {len(odds)} games")
    if odds:
        sample_key = next(iter(odds))
        print(f"Sample game: {sample_key}")
        print(f"Odds: {odds[sample_key]}")