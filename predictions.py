import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import requests
from typing import Dict, List, Optional
import sys
import os
from datetime import datetime, timezone
from dateutil import parser as date_parser
import pytz

# Add the current directory to the path so we can import data_collection
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_collection import fetch_games, fetch_betting_lines, fetch_adjusted_efficiency, fetch_team_stats
    from underdog_value import (
        identify_underdog_value, 
        format_value_bet_display, 
        get_betting_recommendation,
        moneyline_to_implied_probability
    )
    from data_tools.efficiency_loader import EfficiencyDataLoader
    from fetch_live_odds import fetch_live_odds
    from features import find_upset_candidates, predict_win_probability
except ImportError as e:
    st.error(f"Could not import required functions: {e}")
    st.stop()

# Configuration
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"

def get_dataframe_height(df, row_height=35, header_height=38, padding=2, max_height=600):
    """
    Calculate the optimal height for a Streamlit dataframe based on number of rows.
    
    Args:
        df (pd.DataFrame): The dataframe to display
        row_height (int): Height per row in pixels. Default: 35
        header_height (int): Height of header row in pixels. Default: 38
        padding (int): Extra padding in pixels. Default: 2
        max_height (int): Maximum height cap in pixels. Default: 600 (None for no limit)
    
    Returns:
        int: Calculated height in pixels
    
    Example:
        height = get_dataframe_height(my_df)
        st.dataframe(my_df, height=height)
    """
    num_rows = len(df)
    calculated_height = (num_rows * row_height) + header_height + padding
    
    if max_height is not None:
        return min(calculated_height, max_height)
    return calculated_height

# Load models
@st.cache_resource
def load_models():
    """Load trained prediction models."""
    models = {}

    # Spread models
    try:
        models['spread'] = {
            'xgboost': joblib.load(MODEL_DIR / 'spread_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'spread_random_forest.joblib'),
            'linear': joblib.load(MODEL_DIR / 'spread_linear_regression.joblib')
        }
        # Load scaler for linear regression
        try:
            models['spread_scalers'] = {
                'linear': joblib.load(MODEL_DIR / 'spread_linear_regression_scaler.joblib')
            }
        except:
            models['spread_scalers'] = {}
    except:
        st.error("Spread models not found. Please run model training first.")
        models['spread'] = None

    # Total models
    try:
        models['total'] = {
            'xgboost': joblib.load(MODEL_DIR / 'total_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'total_random_forest.joblib'),
            'linear': joblib.load(MODEL_DIR / 'total_linear_regression.joblib')
        }
        # Load scaler for linear regression
        try:
            models['total_scalers'] = {
                'linear': joblib.load(MODEL_DIR / 'total_linear_regression_scaler.joblib')
            }
        except:
            models['total_scalers'] = {}
    except:
        st.error("Total models not found. Please run model training first.")
        models['total'] = None

    # Moneyline models
    try:
        models['moneyline'] = {
            'xgboost': joblib.load(MODEL_DIR / 'moneyline_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'moneyline_random_forest.joblib'),
            'logistic': joblib.load(MODEL_DIR / 'moneyline_logistic_regression.joblib')
        }
        # Load scaler for logistic regression
        try:
            models['moneyline_scalers'] = {
                'logistic': joblib.load(MODEL_DIR / 'moneyline_logistic_regression_scaler.joblib')
            }
        except:
            models['moneyline_scalers'] = {}
    except:
        st.error("Moneyline models not found. Please run model training first.")
        models['moneyline'] = None

    return models

@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_live_odds():
    """Load live betting odds, cached for 1 hour."""
    try:
        return fetch_live_odds()
    except Exception as e:
        st.warning(f"Could not load live odds: {e}")
        return {}

def format_game_datetime(game) -> str:
    """Format game date/time in Eastern Time."""
    try:
        # Get the start date from the game
        if isinstance(game, dict):
            start_date_str = game.get('startDate') or game.get('start_date')
        else:
            start_date_str = getattr(game, 'startDate', None) or getattr(game, 'start_date', None)

        if not start_date_str:
            return "Date TBD"

        # Parse the date (it's in ISO format with timezone)
        if isinstance(start_date_str, str):
            game_datetime = date_parser.parse(start_date_str)
        else:
            game_datetime = start_date_str

        # Convert to Eastern Time
        eastern = pytz.timezone('US/Eastern')
        game_datetime_et = game_datetime.astimezone(eastern)

        # Format nicely
        now = datetime.now(eastern)
        if game_datetime_et.date() == now.date():
            # Today
            return f"Today {game_datetime_et.strftime('%I:%M %p ET')}"
        elif (game_datetime_et - now).days == 1:
            # Tomorrow
            return f"Tomorrow {game_datetime_et.strftime('%I:%M %p ET')}"
        elif game_datetime_et.year == now.year:
            # This year
            return game_datetime_et.strftime('%b %d, %I:%M %p ET')
        else:
            # Different year
            return game_datetime_et.strftime('%b %d, %Y %I:%M %p ET')

    except Exception as e:
        return "Date TBD"

def sort_games_by_date(games: List) -> List:
    """Sort games by start date (most recent first)."""
    def get_game_datetime(game):
        try:
            if isinstance(game, dict):
                start_date_str = game.get('startDate') or game.get('start_date')
            else:
                start_date_str = getattr(game, 'startDate', None) or getattr(game, 'start_date', None)

            if start_date_str:
                return date_parser.parse(start_date_str)
            else:
                # If no date, put at the end
                return datetime.min.replace(tzinfo=timezone.utc)
        except:
            return datetime.min.replace(tzinfo=timezone.utc)

    return sorted(games, key=get_game_datetime, reverse=True)

@st.cache_data(ttl=3600)
def load_espn_games() -> pd.DataFrame:
    """Load ESPN game data from CSV."""
    espn_file = DATA_DIR / "espn_cbb_current_season.csv"
    if espn_file.exists():
        return pd.read_csv(espn_file)
    return pd.DataFrame()

def normalize_team_name(espn_name: str) -> str:
    """Convert ESPN team name to CBBD format.
    ESPN uses 'Michigan Wolverines', CBBD uses 'Michigan'.
    """
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
    
    # Handle special cases
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
        'Southern Illinois Salukis': 'Southern Illinois',
        'Saint Francis Red Flash': 'Saint Francis (PA)',
        'New Haven Chargers': 'New Haven',
        'New Haven': 'New Haven',
        'Stonehill Skyhawks': 'Stonehill',
        'Mercyhurst Lakers': 'Mercyhurst',
        'Wagner Seahawks': 'Wagner',
        'Le Moyne Dolphins': 'Le Moyne',
        'San Francisco Dons': 'San Francisco',
        'USC Trojans': 'USC',
        'Wisconsin Badgers': 'Wisconsin',
        'Michigan State Spartans': 'Michigan State',
        'Rutgers Scarlet Knights': 'Rutgers',
        'Providence Friars': 'Providence',
        'George Washington Revolutionaries': 'George Washington',
        'Massachusetts Minutemen': 'Massachusetts',
        'TCU Horned Frogs': 'TCU',
        'Dayton Flyers': 'Dayton',
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
    
    if espn_name in special_cases:
        return special_cases[espn_name]
    
    # Check for multi-word mascots first
    for mascot in multi_word_mascots:
        if espn_name.endswith(' ' + mascot):
            return espn_name[:-len(' ' + mascot)].strip()
    
    # Remove single-word mascot from end
    parts = espn_name.split()
    if len(parts) > 1 and parts[-1] in mascots:
        return ' '.join(parts[:-1])
    
    return espn_name

# @st.cache_data(ttl=3600)
def get_team_data(season: int = 2025):
    """Fetch team stats and efficiency ratings for specified season with fallback to previous seasons."""
    # Try current season first, then fall back to previous seasons
    for s in [season, season-1, season-2]:
        try:
            efficiency_list = fetch_adjusted_efficiency(s)
            stats_list = fetch_team_stats(s)
            if efficiency_list and stats_list:
                return efficiency_list, stats_list, s
        except:
            continue
    return [], [], None

def get_kenpom_barttorvik_data():
    """Load KenPom and BartTorvik efficiency data for all teams."""
    try:
        loader = EfficiencyDataLoader()
        kenpom_df = loader.load_kenpom()
        bart_df = loader.load_barttorvik()
        return kenpom_df, bart_df
    except Exception as e:
        print(f"Error loading KenPom/BartTorvik data: {e}")
        return None, None

def enrich_with_advanced_metrics(home_team_name, away_team_name, kenpom_df=None, bart_df=None):
    """Enrich team efficiency with KenPom and BartTorvik metrics."""
    metrics = {
        'home': {'kenpom': None, 'barttorvik': None},
        'away': {'kenpom': None, 'barttorvik': None}
    }
    
    if kenpom_df is not None:
        home_kp = kenpom_df[kenpom_df['canonical_team'] == home_team_name]
        away_kp = kenpom_df[kenpom_df['canonical_team'] == away_team_name]
        
        if not home_kp.empty:
            metrics['home']['kenpom'] = {
                'NetRtg': home_kp.iloc[0]['NetRtg'],
                'ORtg': home_kp.iloc[0]['ORtg'],
                'DRtg': home_kp.iloc[0]['DRtg'],
                'AdjT': home_kp.iloc[0]['AdjT'],
                'Luck': home_kp.iloc[0]['Luck'],
                'SOS_NetRtg': home_kp.iloc[0]['SOS_NetRtg']
            }
        
        if not away_kp.empty:
            metrics['away']['kenpom'] = {
                'NetRtg': away_kp.iloc[0]['NetRtg'],
                'ORtg': away_kp.iloc[0]['ORtg'],
                'DRtg': away_kp.iloc[0]['DRtg'],
                'AdjT': away_kp.iloc[0]['AdjT'],
                'Luck': away_kp.iloc[0]['Luck'],
                'SOS_NetRtg': away_kp.iloc[0]['SOS_NetRtg']
            }
    
    if bart_df is not None:
        home_bt = bart_df[bart_df['canonical_team'] == home_team_name]
        away_bt = bart_df[bart_df['canonical_team'] == away_team_name]
        
        if not home_bt.empty:
            row = home_bt.iloc[0]
            # BartTorvik canonical CSV may use different column names; try common names with fallbacks
            adj_oe = row.get('Adj OE') if 'Adj OE' in home_bt.columns else (row.get('Adj_OE') if 'Adj_OE' in home_bt.columns else (row.get('H2') if 'H2' in home_bt.columns else None))
            adj_de = row.get('Adj DE') if 'Adj DE' in home_bt.columns else (row.get('Adj_DE') if 'Adj_DE' in home_bt.columns else (row.get('H3') if 'H3' in home_bt.columns else None))
            metrics['home']['barttorvik'] = {
                'Adj OE': adj_oe,
                'Adj DE': adj_de
            }
        
        if not away_bt.empty:
            row = away_bt.iloc[0]
            adj_oe = row.get('Adj OE') if 'Adj OE' in away_bt.columns else (row.get('Adj_OE') if 'Adj_OE' in away_bt.columns else (row.get('H2') if 'H2' in away_bt.columns else None))
            adj_de = row.get('Adj DE') if 'Adj DE' in away_bt.columns else (row.get('Adj_DE') if 'Adj_DE' in away_bt.columns else (row.get('H3') if 'H3' in away_bt.columns else None))
            metrics['away']['barttorvik'] = {
                'Adj OE': adj_oe,
                'Adj DE': adj_de
            }
    
    return metrics

def enrich_espn_game_with_cbbd_data(game_row, efficiency_list, stats_list, season_used) -> Optional[Dict]:
    """Combine ESPN game data with CBBD stats and efficiency ratings."""
    try:
        home_team_espn = game_row['home_team']
        away_team_espn = game_row['away_team']
        
        # Normalize team names to match CBBD format
        home_team = normalize_team_name(home_team_espn)
        away_team = normalize_team_name(away_team_espn)
        
        # Load live odds
        live_odds = load_live_odds()
        
        # Find matching efficiency and stats (CBBD returns dicts, not objects)
        home_eff = next((e for e in efficiency_list if e.get('team') == home_team), None)
        away_eff = next((e for e in efficiency_list if e.get('team') == away_team), None)
        home_stats_obj = next((s for s in stats_list if s.get('team') == home_team), None)
        away_stats_obj = next((s for s in stats_list if s.get('team') == away_team), None)
        
        # If any data is missing, substitute reasonable defaults so the game can be
        # enriched and passed to the prediction step instead of being skipped.
        def default_eff_from_rank(rank):
            # Create a conservative default offensive/defensive rating based on rank
            try:
                r = int(rank) if rank is not None else None
            except Exception:
                r = None
            if r and r != 99:
                val = 110 - (r / 10)
            else:
                val = 100.1
            return {'offensiveRating': val, 'defensiveRating': val, 'netRating': 0}

        def default_stats():
            # Minimal stats structure matching what extract_stats() expects
            return {
                'games': 32,
                'pace': 70,
                'teamStats': {
                    'points': {'total': 2240},
                    'fourFactors': {
                        'effectiveFieldGoalPct': 48.0,
                        'turnoverRatio': 0.15,
                        'offensiveReboundPct': 30.0,
                        'freeThrowRate': 30.0
                    },
                    'fieldGoals': {'pct': 44.0},
                    'threePointFieldGoals': {'pct': 33.0}
                },
                'opponentStats': {'points': {'total': 2240}}
            }

        # Fill missing efficiency entries with defaults derived from ranks
        if not home_eff:
            home_eff = default_eff_from_rank(game_row.get('home_rank'))
        if not away_eff:
            away_eff = default_eff_from_rank(game_row.get('away_rank'))

        # Fill missing stats objects with minimal defaults
        if not home_stats_obj:
            home_stats_obj = default_stats()
        if not away_stats_obj:
            away_stats_obj = default_stats()
        
        # Create efficiency dicts (we already verified these exist above)
        home_eff_dict = {
            'offensiveRating': home_eff.get('offensiveRating', 100),
            'defensiveRating': home_eff.get('defensiveRating', 100),
            'netRating': home_eff.get('netRating', 0)
        }
        
        away_eff_dict = {
            'offensiveRating': away_eff.get('offensiveRating', 100),
            'defensiveRating': away_eff.get('defensiveRating', 100),
            'netRating': away_eff.get('netRating', 0)
        }
        
        # Create stats dicts (we already verified these exist above)
        def extract_stats(stats_dict):
            team_stats = stats_dict.get('teamStats', {})
            opp_stats = stats_dict.get('opponentStats', {})
            four_factors = team_stats.get('fourFactors', {})
            field_goals = team_stats.get('fieldGoals', {})
            three_pt_fg = team_stats.get('threePointFieldGoals', {})
            games = stats_dict.get('games', 32)
            
            return {
                'ppg': team_stats.get('points', {}).get('total', 2240) / games,
                'pace': stats_dict.get('pace', 70),
                'efg_pct': four_factors.get('effectiveFieldGoalPct', 48.0) / 100.0,
                'to_rate': four_factors.get('turnoverRatio', 0.15),
                'orb_pct': four_factors.get('offensiveReboundPct', 30.0) / 100.0,
                'ft_rate': four_factors.get('freeThrowRate', 30.0) / 100.0,
                'opp_ppg': opp_stats.get('points', {}).get('total', 2240) / games,
                'fg_pct': field_goals.get('pct', 44.0) / 100.0,
                'three_pct': three_pt_fg.get('pct', 33.0) / 100.0
            }
        
        home_stats_dict = extract_stats(home_stats_obj)
        away_stats_dict = extract_stats(away_stats_obj)
        
        # Attempt to fetch betting lines for this game and populate moneylines
        betting_spread = None
        betting_over_under = None
        home_ml = None
        away_ml = None
        try:
            lines_data = fetch_betting_lines(season_used or 2026, 'postseason')
            for line in lines_data:
                if isinstance(line, dict):
                    line_home = line.get('homeTeam') or line.get('home_team') or line.get('home')
                    line_away = line.get('awayTeam') or line.get('away_team') or line.get('away')
                    providers = line.get('lines') or []
                else:
                    line_home = (getattr(line, 'home_team', None) or getattr(line, 'homeTeam', None) or getattr(line, 'home', None))
                    line_away = (getattr(line, 'away_team', None) or getattr(line, 'awayTeam', None) or getattr(line, 'away', None))
                    providers = getattr(line, 'lines', []) or []

                try:
                    lh = str(line_home) if line_home is not None else None
                    la = str(line_away) if line_away is not None else None
                except Exception:
                    lh = line_home
                    la = line_away

                if ((lh == home_team_espn and la == away_team_espn) or
                    (lh == home_team and la == away_team) or
                    (lh == normalize_team_name(home_team_espn) and la == normalize_team_name(away_team_espn))):
                    provider = providers[0] if providers and len(providers) > 0 else None
                    if provider:
                        if isinstance(provider, dict):
                            betting_spread = provider.get('spread') or betting_spread
                            betting_over_under = provider.get('overUnder') or provider.get('over_under') or betting_over_under
                            home_ml = provider.get('homeMoneyline') or provider.get('home_moneyline') or home_ml
                            away_ml = provider.get('awayMoneyline') or provider.get('away_moneyline') or away_ml
                        else:
                            betting_spread = getattr(provider, 'spread', None) or betting_spread
                            betting_over_under = getattr(provider, 'overUnder', None) or betting_over_under
                            home_ml = getattr(provider, 'homeMoneyline', None) or getattr(provider, 'home_moneyline', None) or home_ml
                            away_ml = getattr(provider, 'awayMoneyline', None) or getattr(provider, 'away_moneyline', None) or away_ml
                    else:
                        if isinstance(line, dict):
                            betting_spread = line.get('spread') or betting_spread
                            betting_over_under = line.get('overUnder') or betting_over_under
                            home_ml = line.get('homeMoneyline') or line.get('home_moneyline') or home_ml
                            away_ml = line.get('awayMoneyline') or line.get('away_moneyline') or away_ml
                        else:
                            betting_spread = getattr(line, 'spread', None) or betting_spread
                            betting_over_under = getattr(line, 'overUnder', None) or betting_over_under
                            home_ml = getattr(line, 'homeMoneyline', None) or getattr(line, 'home_moneyline', None) or home_ml
                            away_ml = getattr(line, 'awayMoneyline', None) or getattr(line, 'away_moneyline', None) or away_ml
                    break
        except Exception:
            pass

        # If no moneylines from CFBD, try live odds
        if not home_ml and live_odds:
            game_key = f"{normalize_team_name(home_team_espn)} vs {normalize_team_name(away_team_espn)}"
            # Also try the reverse order in case Odds API has home/away swapped
            reverse_key = f"{normalize_team_name(away_team_espn)} vs {normalize_team_name(home_team_espn)}"

            odds = None
            if game_key in live_odds:
                odds = live_odds[game_key]
            elif reverse_key in live_odds:
                # If found with reversed order, we need to swap the odds too
                odds = live_odds[reverse_key]
                # Swap home/away odds since the teams are swapped
                if odds:
                    odds = odds.copy()
                    # Swap moneyline
                    orig_home_ml = odds.get('home_moneyline')
                    orig_away_ml = odds.get('away_moneyline')
                    odds['home_moneyline'] = orig_away_ml
                    odds['away_moneyline'] = orig_home_ml
                    # Swap spread (flip the sign)
                    orig_home_spread = odds.get('home_spread')
                    orig_away_spread = odds.get('away_spread')
                    if orig_home_spread is not None and orig_away_spread is not None:
                        odds['home_spread'] = -orig_away_spread  # Flip sign
                        odds['away_spread'] = -orig_home_spread  # Flip sign
                        # Keep the odds the same since spread direction changed
                        orig_home_spread_odds = odds.get('home_spread_odds')
                        orig_away_spread_odds = odds.get('away_spread_odds')
                        odds['home_spread_odds'] = orig_away_spread_odds
                        odds['away_spread_odds'] = orig_home_spread_odds

            if odds:
                home_ml = odds.get('home_moneyline')
                away_ml = odds.get('away_moneyline')
                # Optionally update spread/over_under if not set
                if not betting_spread:
                    betting_spread = odds.get('home_spread')
                if not betting_over_under:
                    betting_over_under = odds.get('total_line')

        return {
            'home_team': home_team_espn,  # Use ESPN name for display
            'away_team': away_team_espn,  # Use ESPN name for display
            'home_eff': home_eff_dict,
            'away_eff': away_eff_dict,
            'home_stats': home_stats_dict,
            'away_stats': away_stats_dict,
            'betting_spread': betting_spread,
            'betting_over_under': betting_over_under,
            'home_moneyline': home_ml,
            'away_moneyline': away_ml,
            'home_ml': home_ml,
            'away_ml': away_ml,
            'game_date': game_row.get('date', ''),
            'status': game_row.get('status', ''),
            'venue': game_row.get('venue', ''),
            'neutral_site': game_row.get('neutral_site', False),
            'home_rank': game_row.get('home_rank') if game_row.get('home_rank') != 99 else None,
            'away_rank': game_row.get('away_rank') if game_row.get('away_rank') != 99 else None,
            'season_used': season_used  # Track which season data we used
        }
    except Exception as e:
        return None

def get_upcoming_games() -> List[Dict]:
    """Get games from ESPN data and enrich with CBBD stats for predictions."""
    try:
        # Load ESPN games
        espn_df = load_espn_games()
        
        if espn_df.empty:
            st.warning("No ESPN game data found. Run fetch_espn_cbb_scores.py first.")
            return get_sample_games()
        
        # Get team data from CBBD (uses most recent available season)
        with st.spinner("Fetching team stats from recent seasons..."):
            efficiency_list, stats_list, season_used = get_team_data(2025)
        
        if not efficiency_list or not stats_list:
            st.error("Could not fetch team data from CBBD API.")
            return get_sample_games()
        
        st.success(f"Using {season_used} season data for predictions")
        
        # Filter for upcoming or recent games
        eastern = pytz.timezone('US/Eastern')
        now = datetime.now(eastern)
        
        # Convert date column to datetime
        espn_df['date_dt'] = pd.to_datetime(espn_df['date'])
        
        # Separate upcoming and recent games
        upcoming = espn_df[espn_df['date_dt'] > pd.Timestamp.now(tz='UTC')].copy()
        recent = espn_df[espn_df['date_dt'] <= pd.Timestamp.now(tz='UTC')].copy()
        
        # Prioritize upcoming games, but show recent if none upcoming
        if len(upcoming) > 0:
            st.success(f"Found {len(upcoming)} upcoming games!")
            # Process all upcoming games (no artificial limit)
            games_to_process = upcoming.sort_values('date_dt')
            st.info(f"Processing all {len(upcoming)} upcoming games chronologically")
        else:
            st.info("No upcoming games scheduled. Showing recent games for analysis.")
            games_to_process = recent.sort_values('date_dt', ascending=False).head(20)
        
        # Enrich games with CBBD data
        enriched_games = []
        skipped_count = 0
        for idx, game_row in games_to_process.iterrows():
            enriched = enrich_espn_game_with_cbbd_data(game_row, efficiency_list, stats_list, season_used)
            if enriched:
                enriched_games.append(enriched)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            st.info(f"Skipped {skipped_count} games due to missing team data in {season_used} season")
        
        if not enriched_games:
            st.warning("Could not enrich games with team stats. Using sample data.")
            return get_sample_games()
        
        st.info(f"Prepared {len(enriched_games)} games with team stats for predictions")
        return enriched_games
        
    except Exception as e:
        st.error(f"Error loading games: {e}")
        return get_sample_games()

def get_sample_games() -> List[Dict]:
    """Return sample games for demo purposes."""
    return [
        {
            "home_team": "Duke",
            "away_team": "North Carolina",
            "home_eff": {"offensiveRating": 118.5, "defensiveRating": 89.2, "netRating": 29.3},
            "away_eff": {"offensiveRating": 115.8, "defensiveRating": 91.5, "netRating": 24.3},
            "home_stats": {"ppg": 78.5, "pace": 68.2, "efg_pct": 0.52, "to_rate": 0.15, "orb_pct": 0.32, "ft_rate": 0.35, "opp_ppg": 65.2},
            "away_stats": {"ppg": 75.8, "pace": 67.8, "efg_pct": 0.51, "to_rate": 0.16, "orb_pct": 0.31, "ft_rate": 0.33, "opp_ppg": 68.1},
            "betting_spread": -3.5,
            "betting_over_under": 145.5
        },
        {
            "home_team": "Kansas",
            "away_team": "Texas",
            "home_eff": {"offensiveRating": 122.1, "defensiveRating": 87.8, "netRating": 34.3},
            "away_eff": {"offensiveRating": 119.4, "defensiveRating": 90.1, "netRating": 29.3},
            "home_stats": {"ppg": 82.1, "pace": 69.5, "efg_pct": 0.54, "to_rate": 0.14, "orb_pct": 0.34, "ft_rate": 0.37, "opp_ppg": 63.8},
            "away_stats": {"ppg": 79.4, "pace": 68.9, "efg_pct": 0.53, "to_rate": 0.15, "orb_pct": 0.33, "ft_rate": 0.36, "opp_ppg": 66.2},
            "betting_spread": -4.0,
            "betting_over_under": 148.0
        }
    ]

def format_game_data(game) -> Optional[Dict]:
    """Convert API game object to our expected format."""
    try:
        # Extract team names - handle both dict and object formats
        if isinstance(game, dict):
            # Dict format from API
            home_team = game.get('homeTeam') or game.get('home_team')
            away_team = game.get('awayTeam') or game.get('away_team')
        else:
            # Object format
            home_team = (getattr(game, 'home_team', None) or
                        getattr(game, 'home', None) or
                        getattr(game, 'homeTeam', None))
            away_team = (getattr(game, 'away_team', None) or
                        getattr(game, 'away', None) or
                        getattr(game, 'awayTeam', None))

        if not home_team or not away_team:
            return None

        # Get team names as strings
        if isinstance(home_team, str):
            home_team_name = home_team
        elif hasattr(home_team, 'name'):
            home_team_name = home_team.name
        else:
            home_team_name = str(home_team)

        if isinstance(away_team, str):
            away_team_name = away_team
        elif hasattr(away_team, 'name'):
            away_team_name = away_team.name
        else:
            away_team_name = str(away_team)

        # Try to get efficiency data for both teams
        try:
            efficiency_data = fetch_adjusted_efficiency(2026)  # Current season
            home_eff = next((e for e in efficiency_data if getattr(e, 'team', None) == home_team_name), {})
            away_eff = next((e for e in efficiency_data if getattr(e, 'team', None) == away_team_name), {})
        except:
            home_eff = {}
            away_eff = {}

        # Try to get team stats
        try:
            team_stats_data = fetch_team_stats(2026)
            home_stats = next((s for s in team_stats_data if getattr(s, 'team', None) == home_team_name), {})
            away_stats = next((s for s in team_stats_data if getattr(s, 'team', None) == away_team_name), {})
        except:
            home_stats = {}
            away_stats = {}

        # Try to get betting lines (spread, o/u, moneylines) and parse moneylines
        betting_spread = None
        betting_over_under = None
        home_ml = None
        away_ml = None
        try:
            # Use the season we selected earlier where possible
            lines_data = fetch_betting_lines(2026, "postseason")
            # Find lines for this specific game - try multiple matching criteria
            for line in lines_data:
                # The CFBD response may be a dict (cached JSON) or an object
                if isinstance(line, dict):
                    line_home = line.get('homeTeam') or line.get('home_team') or line.get('home')
                    line_away = line.get('awayTeam') or line.get('away_team') or line.get('away')
                    providers = line.get('lines') or []
                else:
                    line_home = (getattr(line, 'home_team', None) or
                                 getattr(line, 'homeTeam', None) or
                                 getattr(line, 'home', None))
                    line_away = (getattr(line, 'away_team', None) or
                                 getattr(line, 'awayTeam', None) or
                                 getattr(line, 'away', None))
                    providers = getattr(line, 'lines', []) or []

                # Normalize some values to strings for comparison
                try:
                    lh = str(line_home) if line_home is not None else None
                    la = str(line_away) if line_away is not None else None
                except Exception:
                    lh = line_home
                    la = line_away

                # Match either by ESPN display name or normalized canonical name
                if ((lh == home_team_name and la == away_team_name) or
                    (lh == normalize_team_name(home_team_name) and la == normalize_team_name(away_team_name))):
                    # Prefer the first provider entry if present
                    provider = providers[0] if providers and len(providers) > 0 else None
                    if provider:
                        # provider may be dict-like or object-like
                        if isinstance(provider, dict):
                            betting_spread = provider.get('spread') or provider.get('formattedSpread') or betting_spread
                            betting_over_under = provider.get('overUnder') or provider.get('over_under') or betting_over_under
                            home_ml = provider.get('homeMoneyline') or provider.get('home_moneyline') or provider.get('homeML') or home_ml
                            away_ml = provider.get('awayMoneyline') or provider.get('away_moneyline') or provider.get('awayML') or away_ml
                        else:
                            betting_spread = getattr(provider, 'spread', None) or getattr(provider, 'formattedSpread', None) or betting_spread
                            betting_over_under = getattr(provider, 'overUnder', None) or getattr(provider, 'over_under', None) or betting_over_under
                            home_ml = getattr(provider, 'homeMoneyline', None) or getattr(provider, 'home_moneyline', None) or getattr(provider, 'homeML', None) or home_ml
                            away_ml = getattr(provider, 'awayMoneyline', None) or getattr(provider, 'away_moneyline', None) or getattr(provider, 'awayML', None) or away_ml
                    else:
                        # Some cached responses may put moneylines at top level
                        if isinstance(line, dict):
                            home_ml = line.get('homeMoneyline') or line.get('home_moneyline') or home_ml
                            away_ml = line.get('awayMoneyline') or line.get('away_moneyline') or away_ml
                            betting_spread = line.get('spread') or betting_spread
                            betting_over_under = line.get('overUnder') or betting_over_under
                        else:
                            home_ml = getattr(line, 'homeMoneyline', None) or getattr(line, 'home_moneyline', None) or home_ml
                            away_ml = getattr(line, 'awayMoneyline', None) or getattr(line, 'away_moneyline', None) or away_ml
                            betting_spread = getattr(line, 'spread', None) or betting_spread
                            betting_over_under = getattr(line, 'overUnder', None) or betting_over_under
                    # Stop after first match
                    break
        except Exception:
            # Do not fail enrichment if betting lines are unavailable
            pass

        # Get actual scores if available
        if isinstance(game, dict):
            actual_home_score = game.get('homePoints') or game.get('home_points')
            actual_away_score = game.get('awayPoints') or game.get('away_points')
        else:
            actual_home_score = getattr(game, 'home_points', None) or getattr(game, 'homePoints', None)
            actual_away_score = getattr(game, 'away_points', None) or getattr(game, 'awayPoints', None)

        # Create the game data structure with default values if data is missing
        return {
            "home_team": home_team_name,
            "away_team": away_team_name,
            "actual_home_score": actual_home_score,
            "actual_away_score": actual_away_score,
            "home_eff": {
                "offensiveRating": float(getattr(home_eff, 'offensive_rating', 100) or getattr(home_eff, 'adj_offense', 100) or 100),
                "defensiveRating": float(getattr(home_eff, 'defensive_rating', 100) or getattr(home_eff, 'adj_defense', 100) or 100),
                "netRating": float(getattr(home_eff, 'net_rating', 0) or getattr(home_eff, 'adj_net', 0) or 0)
            },
            "away_eff": {
                "offensiveRating": float(getattr(away_eff, 'offensive_rating', 100) or getattr(away_eff, 'adj_offense', 100) or 100),
                "defensiveRating": float(getattr(away_eff, 'defensive_rating', 100) or getattr(away_eff, 'adj_defense', 100) or 100),
                "netRating": float(getattr(away_eff, 'net_rating', 0) or getattr(away_eff, 'adj_net', 0) or 0)
            },
            "home_stats": {
                "ppg": float(getattr(home_stats, 'points_per_game', 70) or getattr(home_stats, 'ppg', 70) or 70),
                "pace": float(getattr(home_stats, 'pace', 68) or 68),
                "efg_pct": float(getattr(home_stats, 'effective_fg_pct', 0.50) or getattr(home_stats, 'efg_pct', 0.50) or 0.50),
                "to_rate": float(getattr(home_stats, 'turnover_pct', 0.15) or getattr(home_stats, 'to_rate', 0.15) or 0.15),
                "orb_pct": float(getattr(home_stats, 'orb_pct', 0.30) or 0.30),
                "ft_rate": float(getattr(home_stats, 'ft_rate', 0.35) or 0.35),
                "opp_ppg": float(getattr(home_stats, 'opp_points_per_game', 70) or getattr(home_stats, 'opp_ppg', 70) or 70)
            },
            "away_stats": {
                "ppg": float(getattr(away_stats, 'points_per_game', 70) or getattr(away_stats, 'ppg', 70) or 70),
                "pace": float(getattr(away_stats, 'pace', 68) or 68),
                "efg_pct": float(getattr(away_stats, 'effective_fg_pct', 0.50) or getattr(away_stats, 'efg_pct', 0.50) or 0.50),
                "to_rate": float(getattr(away_stats, 'turnover_pct', 0.15) or getattr(away_stats, 'to_rate', 0.15) or 0.15),
                "orb_pct": float(getattr(away_stats, 'orb_pct', 0.30) or 0.30),
                "ft_rate": float(getattr(away_stats, 'ft_rate', 0.35) or 0.35),
                "opp_ppg": float(getattr(away_stats, 'opp_points_per_game', 70) or getattr(away_stats, 'opp_ppg', 70) or 70)
            },
            "betting_spread": betting_spread,
            "betting_over_under": betting_over_under,
            # Moneyline fields for market comparisons (aliases for compatibility)
            "home_moneyline": home_ml,
            "away_moneyline": away_ml,
            "home_ml": home_ml,
            "away_ml": away_ml
        }

    except Exception as e:
        print(f"Error formatting game {game}: {e}")
        return None

def calculate_features(home_team: Dict, away_team: Dict, home_eff: Dict, away_eff: Dict, advanced_metrics: Dict = None) -> Dict:
    """Calculate prediction features for a game."""
    # Prefer centralized feature helpers from features.py when available
    try:
        from features import (
            calculate_efficiency_differential as _calc_eff_diff,
            calculate_spread_features as _calc_spread_feats,
            calculate_total_features as _calc_total_feats,
            calculate_win_probability_features as _calc_win_feats,
        )

        eff = _calc_eff_diff(home_eff or {}, away_eff or {})
        off_eff_diff = eff.get('off_eff_diff', 0)
        def_eff_diff = eff.get('def_eff_diff', 0)
        net_eff_diff = eff.get('net_eff_diff', 0)
    except Exception:
        # Efficiency features (CBBD - these are the original 3 features)
        off_eff_diff = home_eff.get("offensiveRating", 0) - away_eff.get("offensiveRating", 0)
        def_eff_diff = home_eff.get("defensiveRating", 0) - away_eff.get("defensiveRating", 0)
        net_eff_diff = home_eff.get("netRating", 0) - away_eff.get("netRating", 0)

    # Advanced KenPom features (if available)
    kenpom_features = []
    if advanced_metrics and advanced_metrics.get('home', {}).get('kenpom') and advanced_metrics.get('away', {}).get('kenpom'):
        home_kp = advanced_metrics['home']['kenpom']
        away_kp = advanced_metrics['away']['kenpom']
        
        kenpom_features = [
            home_kp['NetRtg'] - away_kp['NetRtg'],  # Net rating diff
            home_kp['ORtg'] - away_kp['ORtg'],      # Offensive rating diff
            home_kp['DRtg'] - away_kp['DRtg'],      # Defensive rating diff (negative is better for home)
            home_kp['AdjT'] - away_kp['AdjT'],      # Tempo diff
            home_kp['Luck'] - away_kp['Luck'],      # Luck diff
            home_kp['SOS_NetRtg'] - away_kp['SOS_NetRtg']  # SOS diff
        ]
    
    # Advanced BartTorvik features (if available)
    bart_features = []
    if advanced_metrics and advanced_metrics.get('home', {}).get('barttorvik') and advanced_metrics.get('away', {}).get('barttorvik'):
        home_bt = advanced_metrics['home']['barttorvik']
        away_bt = advanced_metrics['away']['barttorvik']
        
        bart_features = [
            home_bt['Adj OE'] - away_bt['Adj OE'],  # Offensive efficiency diff
            home_bt['Adj DE'] - away_bt['Adj DE']   # Defensive efficiency diff
        ]

    # Stats and extended features: try to compute via helpers, otherwise compute inline
    try:
        spread_feats = _calc_spread_feats(home_team or {}, away_team or {}, home_eff or {}, away_eff or {})
        ppg_diff = spread_feats.get('ppg_diff', 0)
        opp_ppg_diff = spread_feats.get('opp_ppg_diff', 0)
        margin_diff = spread_feats.get('margin_diff', 0)
        efg_diff = spread_feats.get('efg_diff', 0)
        to_rate_diff = spread_feats.get('to_rate_diff', 0)
        orb_diff = spread_feats.get('orb_diff', 0)
        ft_rate_diff = spread_feats.get('ft_rate_diff', 0)

        total_feats = _calc_total_feats(home_team or {}, away_team or {}, home_eff or {}, away_eff or {})
        combined_off_eff = total_feats.get('combined_off_eff', 0)
        combined_def_eff = total_feats.get('combined_def_eff', 0)
        avg_off_eff = total_feats.get('avg_off_eff', 0)
        avg_def_eff = total_feats.get('avg_def_eff', 0)
        combined_tempo = total_feats.get('combined_tempo', (home_team or {}).get('pace', 70) + (away_team or {}).get('pace', 70))
        avg_tempo = total_feats.get('avg_tempo', ((home_team or {}).get('pace', 70) + (away_team or {}).get('pace', 70)) / 2)
        combined_ppg = total_feats.get('combined_ppg', (home_team or {}).get('ppg', 0) + (away_team or {}).get('ppg', 0))
        combined_opp_ppg = total_feats.get('combined_opp_ppg', (home_team or {}).get('opp_ppg', 0) + (away_team or {}).get('opp_ppg', 0))
        combined_fg_pct = total_feats.get('combined_fg_pct', (home_team or {}).get('fg_pct', 0) + (away_team or {}).get('fg_pct', 0))
        combined_3pt_pct = total_feats.get('combined_3pt_pct', (home_team or {}).get('three_pct', 0) + (away_team or {}).get('three_pct', 0))
        projected_total = total_feats.get('projected_total')
    except Exception:
        ppg_diff = home_team.get("ppg", 0) - away_team.get("ppg", 0)
        opp_ppg_diff = home_team.get("opp_ppg", 0) - away_team.get("opp_ppg", 0)
        margin_diff = (home_team.get("ppg", 0) - home_team.get("opp_ppg", 0)) - (away_team.get("ppg", 0) - away_team.get("opp_ppg", 0))

        efg_diff = home_team.get("efg_pct", 0) - away_team.get("efg_pct", 0)
        to_rate_diff = home_team.get("to_rate", 0) - away_team.get("to_rate", 0)
        orb_diff = home_team.get("orb_pct", 0) - away_team.get("orb_pct", 0)
        ft_rate_diff = home_team.get("ft_rate", 0) - away_team.get("ft_rate", 0)

        combined_off_eff = home_eff.get("offensiveRating", 0) + away_eff.get("offensiveRating", 0)
        combined_def_eff = home_eff.get("defensiveRating", 0) + away_eff.get("defensiveRating", 0)
        avg_off_eff = (home_eff.get("offensiveRating", 0) + away_eff.get("offensiveRating", 0)) / 2
        avg_def_eff = (home_eff.get("defensiveRating", 0) + away_eff.get("defensiveRating", 0)) / 2
        combined_tempo = (home_team.get("pace", 0) + away_team.get("pace", 0))
        avg_tempo = (home_team.get("pace", 0) + away_team.get("pace", 0)) / 2
        combined_ppg = home_team.get("ppg", 0) + away_team.get("ppg", 0)
        combined_opp_ppg = home_team.get("opp_ppg", 0) + away_team.get("opp_ppg", 0)
        combined_fg_pct = home_team.get("fg_pct", 0.44) + away_team.get("fg_pct", 0.44)
        combined_3pt_pct = home_team.get("three_pct", 0.33) + away_team.get("three_pct", 0.33)
        projected_total = (avg_off_eff + avg_def_eff) / 2 * (avg_tempo / 100) * 0.8
    avg_def_eff = (home_eff.get("defensiveRating", 0) + away_eff.get("defensiveRating", 0)) / 2

    combined_tempo = home_team.get("pace", 0) + away_team.get("pace", 0)
    avg_tempo = (home_team.get("pace", 0) + away_team.get("pace", 0)) / 2
    combined_ppg = home_team.get("ppg", 0) + away_team.get("ppg", 0)
    combined_opp_ppg = home_team.get("opp_ppg", 0) + away_team.get("opp_ppg", 0)
    combined_fg_pct = home_team.get("fg_pct", 0.44) + away_team.get("fg_pct", 0.44)
    combined_3pt_pct = home_team.get("three_pct", 0.33) + away_team.get("three_pct", 0.33)

    # Projected total
    projected_total = (avg_off_eff + avg_def_eff) / 2 * (avg_tempo / 100) * 0.8

    # Base features (original 3 used in current models)
    base_features = [off_eff_diff, def_eff_diff, net_eff_diff]
    
    # Extended features for model predictions
    extended_features = base_features.copy()
    
    # Add KenPom features if available
    if len(kenpom_features) > 0:
        extended_features.extend(kenpom_features)
    else:
        # Add zeros as placeholders if KenPom not available
        extended_features.extend([0, 0, 0, 0, 0, 0])
    
    # Add BartTorvik features if available
    if len(bart_features) > 0:
        extended_features.extend(bart_features)
    else:
        # Add zeros as placeholders if BartTorvik not available
        extended_features.extend([0, 0])

    return {
        # Use extended features for all predictions (11 features total)
        'spread': extended_features,
        'total': extended_features,
        'moneyline': extended_features,
        
        # Metadata
        'kenpom_available': len(kenpom_features) > 0,
        'barttorvik_available': len(bart_features) > 0
    }

def make_predictions(game_data: Dict, models: Dict, advanced_metrics: Dict = None) -> Dict:
    """Make predictions for a game using trained models."""
    features = calculate_features(
        game_data['home_stats'], game_data['away_stats'],
        game_data['home_eff'], game_data['away_eff'],
        advanced_metrics
    )

    predictions = {}
    
    # Store advanced metrics availability
    predictions['advanced_metrics_available'] = features.get('kenpom_available', False) or features.get('barttorvik_available', False)

    # Define feature names that match training data (11 features)
    feature_names = ['off_eff_diff', 'def_eff_diff', 'net_eff_diff',
                     'kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff',
                     'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff',
                     'bart_oe_diff', 'bart_de_diff']

    # Spread predictions
    if models.get('spread'):
        spread_preds = []
        # Convert to DataFrame with proper column names
        spread_df = pd.DataFrame([features['spread']], columns=feature_names)
        scalers = models.get('spread_scalers', {})
        
        for model_name, model in models['spread'].items():
            try:
                # Apply scaler if available (for linear regression)
                if model_name in scalers:
                    scaled_df = scalers[model_name].transform(spread_df)
                    pred = model.predict(scaled_df)[0]
                else:
                    pred = model.predict(spread_df)[0]
                spread_preds.append(pred)
            except Exception as e:
                print(f"Error predicting spread with {model_name}: {e}")

        if spread_preds:
            predictions['spread'] = {
                'prediction': np.mean(spread_preds),
                'range': f"{min(spread_preds):.1f} to {max(spread_preds):.1f}",
                'models': spread_preds
            }

    # Total predictions
    if models.get('total'):
        total_preds = []
        # Convert to DataFrame with proper column names
        total_df = pd.DataFrame([features['total']], columns=feature_names)
        scalers = models.get('total_scalers', {})
        
        for model_name, model in models['total'].items():
            try:
                # Apply scaler if available (for linear regression)
                if model_name in scalers:
                    scaled_df = scalers[model_name].transform(total_df)
                    pred = model.predict(scaled_df)[0]
                else:
                    pred = model.predict(total_df)[0]
                total_preds.append(pred)
            except Exception as e:
                print(f"Error predicting total with {model_name}: {e}")

        if total_preds:
            predictions['total'] = {
                'prediction': np.mean(total_preds),
                'range': f"{min(total_preds):.1f} to {max(total_preds):.1f}",
                'models': total_preds
            }

    # Moneyline predictions
    if models.get('moneyline'):
        moneyline_preds = []
        # Use same features as spread
        moneyline_df = pd.DataFrame([features['moneyline']], columns=feature_names)
        scalers = models.get('moneyline_scalers', {})
        
        for model_name, model in models['moneyline'].items():
            try:
                # Apply scaler if available (for logistic regression)
                if model_name in scalers:
                    scaled_df = scalers[model_name].transform(moneyline_df)
                    pred_proba = model.predict_proba(scaled_df)[0]
                else:
                    pred_proba = model.predict_proba(moneyline_df)[0]
                home_win_prob = pred_proba[1]  # Probability of home win (class 1)
                moneyline_preds.append(home_win_prob)
            except Exception as e:
                print(f"Error predicting moneyline with {model_name}: {e}")
                # Skip this model if it fails
                continue

        if moneyline_preds:
            avg_prob = np.mean(moneyline_preds)
            max_prob = max(avg_prob, 1 - avg_prob)
            # Predicted probability: probability for the predicted side (Home or Away)
            predicted_prob = avg_prob if avg_prob > 0.5 else (1 - avg_prob)

            predictions['moneyline'] = {
                'home_win_prob': avg_prob,
                'away_win_prob': 1 - avg_prob,
                'prediction': 'Home' if avg_prob > 0.5 else 'Away',
                # Keep the key `confidence` for compatibility, but store raw predicted probability.
                'confidence': f"{predicted_prob:.1%}",
                'models': moneyline_preds
            }

    return predictions

def render_game_prediction(game: Dict, predictions: Dict, efficiency_list: List = None, stats_list: List = None, models: Dict = None):
    """Render a game prediction card."""
    # Show game header with rankings if available
    home_rank_str = f" (#{int(game['home_rank'])})" if game.get('home_rank') else ""
    away_rank_str = f" (#{int(game['away_rank'])})" if game.get('away_rank') else ""
    
    st.subheader(f"🏀 {game['away_team']}{away_rank_str} @ {game['home_team']}{home_rank_str}")
    
    # Show which season's data is being used
    if game.get('season_used'):
        st.caption(f"📊 Using {game['season_used']} season statistics")
    
    # Show game details
    col1, col2, col3 = st.columns(3)
    with col1:
        if game.get('game_date'):
            try:
                game_dt = pd.to_datetime(game['game_date'])
                # Check if time is midnight (00:00:00), which means time is TBD
                if game_dt.hour == 0 and game_dt.minute == 0:
                    st.caption(f"📅 {game_dt.strftime('%a, %b %d, %Y')} (Time TBD)")
                else:
                    st.caption(f"📅 {game_dt.strftime('%a, %b %d, %Y %I:%M %p ET')}")
            except:
                st.caption(f"📅 {game.get('game_date', 'TBD')}")
        else:
            st.caption("📅 Date TBD")
    with col2:
        if game.get('venue'):
            neutral = " (Neutral)" if game.get('neutral_site') else ""
            st.caption(f"📍 {game['venue']}{neutral}")
    with col3:
        if game.get('status'):
            status = game['status']
            if status == 'STATUS_SCHEDULED':
                st.caption("⏱️ Upcoming")
            elif status == 'STATUS_FINAL':
                st.caption("✅ Final")
            else:
                st.caption(f"📊 {status}")

    # Show actual score if available
    actual_home = game.get('actual_home_score')
    actual_away = game.get('actual_away_score')

    if actual_home is not None and actual_away is not None:
        actual_spread = actual_home - actual_away
        actual_total = actual_home + actual_away

        st.markdown(f"**Final Score:** {game['away_team']} {actual_away} - {game['home_team']} {actual_home}")
        st.markdown(f"**Actual Spread:** {actual_spread:+.0f} | **Actual Total:** {actual_total:.0f}")
        st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Spread Prediction**")
        if 'spread' in predictions:
            pred = predictions['spread']
            betting_spread = game.get('betting_spread')
            
            # Format spread with team name
            spread_val = pred['prediction']
            if spread_val < 0:
                spread_display = f"{game['home_team']} {spread_val:.1f}"
            else:
                spread_display = f"{game['away_team']} +{spread_val:.1f}"

            st.metric(
                label="Predicted Spread" + (f" (Line: {betting_spread:.1f})" if betting_spread else ""),
                value=spread_display,
                delta=f"{abs(pred['prediction'] - betting_spread):.1f} pts difference" if betting_spread else None
            )

            if actual_home is not None and actual_away is not None:
                actual_spread = actual_home - actual_away
                spread_diff = pred['prediction'] - actual_spread
                if abs(spread_diff) <= 3:
                    st.success(f"✅ Accurate! (Off by {abs(spread_diff):.1f} pts)")
                elif abs(spread_diff) <= 7:
                    st.warning(f"⚠️ Close (Off by {abs(spread_diff):.1f} pts)")
                else:
                    st.error(f"❌ Missed (Off by {abs(spread_diff):.1f} pts)")

            if betting_spread and abs(pred['prediction'] - betting_spread) > 3:
                st.success("🎯 Potential Value Bet!")
            elif betting_spread and abs(pred['prediction'] - betting_spread) < 1:
                st.info("📊 Aligned with line")

    with col2:
        st.markdown("**Total Prediction (Combined Score)**")
        if 'total' in predictions:
            pred = predictions['total']
            betting_total = game.get('betting_over_under')
            
            # Ensure total is positive (it's a combined score)
            total_val = abs(pred['prediction'])

            st.metric(
                label="Total Points (Both Teams)" + (f" - O/U Line: {betting_total:.1f}" if betting_total else ""),
                value=f"{total_val:.1f} pts",
                delta=f"{total_val - betting_total:.1f} pts difference" if betting_total else None
            )

            if actual_home is not None and actual_away is not None:
                actual_total = actual_home + actual_away
                total_diff = total_val - actual_total
                if abs(total_diff) <= 5:
                    st.success(f"✅ Accurate! (Off by {abs(total_diff):.1f} pts)")
                elif abs(total_diff) <= 10:
                    st.warning(f"⚠️ Close (Off by {abs(total_diff):.1f} pts)")
                else:
                    st.error(f"❌ Missed (Off by {abs(total_diff):.1f} pts)")

    with col3:
        st.markdown("**Moneyline Prediction**")
        if 'moneyline' in predictions:
            pred = predictions['moneyline']

            # Show full probability column: both home and away win probabilities
            st.metric(
                label=f"{game['home_team']} Win Probability",
                value=f"{pred.get('home_win_prob', 0):.1%}"
            )
            st.metric(
                label=f"{game['away_team']} Win Probability",
                value=f"{pred.get('away_win_prob', 0):.1%}"
            )

            # Show market implied probabilities and edge if betting moneylines are available
            try:
                from features import calculate_implied_probability as _implied

                home_ml = game.get('home_moneyline') or game.get('home_ml')
                away_ml = game.get('away_moneyline') or game.get('away_ml')
                if home_ml is not None and away_ml is not None:
                    implied_home = _implied(home_ml)
                    implied_away = _implied(away_ml)
                    st.caption(f"Market Implied — {game['home_team']}: {implied_home:.1%}, {game['away_team']}: {implied_away:.1%}")
                    # Edge for favored side if model probabilities exist
                    if pred.get('prediction') == 'Home' and pred.get('home_win_prob') is not None:
                        edge = pred.get('home_win_prob') - implied_home
                        st.caption(f"Edge: {edge:.1%} (model - market)")
                    elif pred.get('prediction') == 'Away' and pred.get('away_win_prob') is not None:
                        edge = pred.get('away_win_prob') - implied_away
                        st.caption(f"Edge: {edge:.1%} (model - market)")
                elif home_ml is not None or away_ml is not None:
                    ml = home_ml if home_ml is not None else away_ml
                    implied = _implied(ml)
                    st.caption(f"Market Implied: {implied:.1%}")
                    if pred.get('prediction') == 'Home' and pred.get('home_win_prob') is not None:
                        edge = pred.get('home_win_prob') - implied
                        st.caption(f"Edge: {edge:.1%} (model - market)")
                    elif pred.get('prediction') == 'Away' and pred.get('away_win_prob') is not None:
                        edge = pred.get('away_win_prob') - implied
                        st.caption(f"Edge: {edge:.1%} (model - market)")
            except Exception:
                pass

            if actual_home is not None and actual_away is not None:
                actual_winner = 'Home' if actual_home > actual_away else 'Away'
                if pred['prediction'] == actual_winner:
                    st.success("✅ Correct winner prediction!")
                else:
                    st.error("❌ Wrong winner prediction")

            st.caption(f"Predicted Probability: {pred['confidence']}")
    
    # Check for underdog value bets
    if 'moneyline' in predictions and game.get('home_moneyline') and game.get('away_moneyline'):
        value_bet = identify_underdog_value(
            game,
            predictions['moneyline']['home_win_prob'],
            min_ev_threshold=5.0
        )
        
        if value_bet:
            st.divider()
            st.markdown("### 🎯 VALUE BET DETECTED!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**{value_bet['team']}** (Underdog)")
                st.metric("Moneyline", f"{value_bet['moneyline']:+d}")
                st.metric("Model Probability", f"{value_bet['model_prob']:.1%}")
                st.metric("Implied Probability", f"{value_bet['implied_prob']:.1%}")
            
            with col2:
                st.metric("Edge", f"{value_bet['edge']:.1%}", delta="Positive")
                st.metric("Expected Value (per $100)", f"${value_bet['expected_value']:.2f}")
                st.metric("ROI", f"{value_bet['roi']:.1f}%", delta="Positive")
            
            # Betting recommendation
            with st.expander("📊 Betting Recommendation"):
                bankroll = st.number_input("Your Bankroll ($)", min_value=100, max_value=100000, value=1000, step=100)
                kelly_fraction = st.slider("Kelly Fraction (Conservative: 0.25)", 0.1, 1.0, 0.25, 0.05)
                
                recommendation = get_betting_recommendation(value_bet, bankroll, kelly_fraction)
                
                st.markdown(f"""
                **Recommended Bet Size**: ${recommendation['recommended_bet']:.2f} ({recommendation['kelly_percentage']:.1f}% of bankroll)
                
                **Potential Outcomes**:
                - Risk: ${recommendation['risk_amount']:.2f}
                - Win: ${recommendation['potential_profit']:.2f} profit
                - Total Return: ${recommendation['potential_return']:.2f}
                
                💡 *Using fractional Kelly criterion for risk management*
                """)             

    # Check for upset potential (for tournament games)
    if 'moneyline' in predictions and game.get('home_rank') and game.get('away_rank'):
        # Determine favorite and underdog based on rankings
        home_rank = game.get('home_rank', 99)
        away_rank = game.get('away_rank', 99)
        
        if home_rank and away_rank and home_rank != away_rank:
            # Lower rank number = better team
            favorite_rank = min(home_rank, away_rank)
            underdog_rank = max(home_rank, away_rank)
            rank_diff = underdog_rank - favorite_rank
            
            # Only check for significant rank differences (like seed differences)
            if rank_diff >= 10:  # Equivalent to about 3-4 seed difference
                favorite_team = game['home_team'] if home_rank == favorite_rank else game['away_team']
                underdog_team = game['away_team'] if home_rank == favorite_rank else game['home_team']
                
                # Get moneyline for underdog
                underdog_ml = game.get('away_moneyline') if underdog_team == game['away_team'] else game.get('home_moneyline')
                
                if underdog_ml:
                    # Create matchup data for upset detection
                    matchup = {
                        'higher_seed_team': favorite_team,
                        'lower_seed_team': underdog_team,
                        'higher_seed': favorite_rank,
                        'lower_seed': underdog_rank,
                        'underdog_ml': underdog_ml
                    }
                    
                    # Create efficiency and stats lookups
                    efficiency_lookup = {team['team']: team for team in efficiency_list}
                    stats_lookup = {team['team']: team for team in stats_list}
                    
                    # Find upset candidates
                    upsets = find_upset_candidates(
                        [matchup], 
                        min_seed_diff=rank_diff,
                        efficiency_data=efficiency_lookup,
                        stats_data=stats_lookup,
                        models=models
                    )
                    
                    if upsets:
                        upset = upsets[0]  # Should only be one
                        st.divider()
                        st.markdown("### 🔥 UPSET ALERT!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"**{upset['underdog']}** (Underdog)")
                            st.markdown(f"vs **{upset['favorite']}** (Favorite)")
                            st.metric("Rank Matchup", upset['seed_matchup'])
                            st.metric("Moneyline", f"{upset['moneyline']:+d}")
                        
                        with col2:
                            st.metric("Upset Probability", f"{upset['upset_prob']:.1%}")
                            st.metric("Implied Probability", f"{upset['implied_prob']:.1%}")
                            st.metric("Edge", f"{upset['edge']:.1%}", delta="Positive" if upset['edge'] > 0 else "normal")
                        
                        st.info("💡 This underdog has a strong chance of pulling off an upset based on efficiency metrics!")


def main():
    st.set_page_config(
        page_title="Bracket Oracle - March Madness Predictions",
        page_icon="🏀",
        layout="wide"
    )

    # Logo
    st.image("data_files/logo.png", width=250)

    st.title("🏀 Bracket Oracle - March Madness Predictions")
    # st.markdown("*AI-powered betting predictions using efficiency ratings and team statistics*")

    # Load models
    models = load_models()
    
    # Load advanced efficiency metrics
    kenpom_df, bart_df = get_kenpom_barttorvik_data()
    if kenpom_df is not None:
        st.sidebar.success(f"✓ KenPom data loaded ({len(kenpom_df)} teams)")
    if bart_df is not None:
        st.sidebar.success(f"✓ BartTorvik data loaded ({len(bart_df)} teams)")
    
    # Fetch team data for predictions
    efficiency_list, stats_list, season_used = get_team_data()
    if efficiency_list and stats_list:
        st.sidebar.success(f"✓ Team data loaded (season {season_used})")
    else:
        st.sidebar.error("❌ Could not load team data for predictions")
        st.error("Unable to load team statistics. Predictions cannot be generated.")
        return

    # Sidebar
    st.sidebar.header("Model Performance")
    
    # Load real metrics from training
    import json
    spread_metrics = {}
    total_metrics = {}
    moneyline_metrics = {}
    
    try:
        with open(MODEL_DIR / "spread_metrics.json", 'r') as f:
            spread_metrics = json.load(f)
    except:
        pass
    
    try:
        with open(MODEL_DIR / "total_metrics.json", 'r') as f:
            total_metrics = json.load(f)
    except:
        pass
    
    try:
        with open(MODEL_DIR / "moneyline_metrics.json", 'r') as f:
            moneyline_metrics = json.load(f)
    except:
        pass
    
    # Display metrics with real values or fallbacks
    spread_mae = spread_metrics.get('mae', 11.25)
    total_mae = total_metrics.get('mae', 11.58)
    moneyline_acc = moneyline_metrics.get('accuracy', 0.711)
    
    st.sidebar.metric("Spread MAE", f"{spread_mae:.2f} pts")
    st.sidebar.metric("Total MAE", f"{total_mae:.2f} pts")
    st.sidebar.metric("Moneyline Accuracy", f"{moneyline_acc:.1%}")
    
    # Show metric ranges if available
    if spread_metrics.get('mae_range'):
        st.sidebar.caption(f"Spread range: {spread_metrics['mae_range']}")
    if total_metrics.get('mae_range'):
        st.sidebar.caption(f"Total range: {total_metrics['mae_range']}")
    if moneyline_metrics.get('accuracy_range'):
        st.sidebar.caption(f"Moneyline range: {moneyline_metrics['accuracy_range']}")

    st.sidebar.header("Data Source")
    st.sidebar.info("Using 2025 season statistics. Predictions based on team performance patterns from historical data.")

    # Get upcoming games
    games = get_upcoming_games()

    if not games:
        st.error("No games available. Please check your API connection and try again.")
        return

    # Sort games by date (most recent first)
    games = sort_games_by_date(games)

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 All Games Table", "🎯 Individual Game Analysis", "🎲 Parlay Builder", "📈 Historical Against the Spread"])

    with tab1:
        st.header("📊 All Games with Predictions")
        st.markdown("Complete table of all upcoming games with AI-powered betting predictions.")

        # Create table data
        table_data = []
        for game in games:
            # Enrich game with CBBD data
            enriched_game = enrich_espn_game_with_cbbd_data(game, efficiency_list, stats_list, season_used)
            if not enriched_game:
                # Skip games we can't enrich
                continue
            
            # Get advanced metrics for this game
            advanced_metrics = None
            if kenpom_df is not None or bart_df is not None:
                home_team = normalize_team_name(game['home_team'])
                away_team = normalize_team_name(game['away_team'])
                advanced_metrics = enrich_with_advanced_metrics(home_team, away_team, kenpom_df, bart_df)
            
            # Make predictions for this game
            try:
                predictions = make_predictions(enriched_game, models, advanced_metrics)
            except Exception as e:
                st.warning(f"Could not generate predictions for {game['away_team']} @ {game['home_team']}: {e}")
                predictions = {}

            # Format date
            if game.get('game_date'):
                try:
                    game_dt = pd.to_datetime(game['game_date'])
                    if game_dt.hour == 0 and game_dt.minute == 0:
                        date_str = game_dt.strftime('%a, %b %d (Time TBD)')
                    else:
                        date_str = game_dt.strftime('%a, %b %d %I:%M %p')
                except:
                    date_str = "Date TBD"
            else:
                date_str = "Date TBD"

            # Format rankings
            away_rank = f"(#{int(game['away_rank'])})" if game.get('away_rank') else ""
            home_rank = f"(#{int(game['home_rank'])})" if game.get('home_rank') else ""

            # Format predictions
            moneyline_pred = predictions.get('moneyline', {})
            spread_pred = predictions.get('spread', {})
            total_pred = predictions.get('total', {})

            moneyline_str = f"{moneyline_pred.get('prediction', 'N/A')} ({moneyline_pred.get('confidence', 'N/A')})" if moneyline_pred else "N/A"
            
            # Format spread to show which team has the line
            if spread_pred and 'prediction' in spread_pred:
                spread_val = spread_pred['prediction']
                if spread_val < 0:
                    spread_str = f"Home {spread_val:+.1f}"
                else:
                    spread_str = f"Away {spread_val:+.1f}"
            else:
                spread_str = "N/A"
                
            total_str = f"{total_pred.get('prediction', 'N/A'):.1f}" if total_pred else "N/A"

            # Model probability, market implied probability (if available), and edge
            pred_model_str = "N/A"
            market_prob_str = "N/A"
            edge_str = "N/A"

            if moneyline_pred:
                try:
                    model_home = float(moneyline_pred.get('home_win_prob', 0) or 0)
                    model_away = float(moneyline_pred.get('away_win_prob', 0) or 0)
                    # Favored side according to model
                    favored = moneyline_pred.get('prediction')
                    if favored == 'Home':
                        pred_model_str = f"Home {model_home:.1%}"
                    else:
                        pred_model_str = f"Away {model_away:.1%}"
                except Exception:
                    pred_model_str = moneyline_pred.get('confidence', 'N/A')

            # Try to compute market/implied probabilities from available moneyline odds
            try:
                from features import calculate_implied_probability as _implied

                home_ml = enriched_game.get('home_moneyline') or enriched_game.get('home_ml') or None
                away_ml = enriched_game.get('away_moneyline') or enriched_game.get('away_ml') or None
                if home_ml is not None and away_ml is not None:
                    implied_home = _implied(home_ml)
                    implied_away = _implied(away_ml)
                    if favored == 'Home':
                        market_prob_str = f"Home {implied_home:.1%}"
                        if pred_model_str != 'N/A':
                            edge_str = f"{(model_home - implied_home):.1%}"
                    else:
                        market_prob_str = f"Away {implied_away:.1%}"
                        if pred_model_str != 'N/A':
                            edge_str = f"{(model_away - implied_away):.1%}"
                else:
                    # If only a single moneyline present (home_ml), show implied for whichever exists
                    ml = home_ml if home_ml is not None else away_ml
                    if ml is not None:
                        implied = _implied(ml)
                        # If model favors home, assume ml applies to home
                        if favored == 'Home':
                            market_prob_str = f"Home {implied:.1%}"
                            if pred_model_str != 'N/A':
                                edge_str = f"{(model_home - implied):.1%}"
                        else:
                            market_prob_str = f"Away {implied:.1%}"
                            if pred_model_str != 'N/A':
                                edge_str = f"{(model_away - implied):.1%}"
            except Exception:
                pass

            # Check for upset potential
            upset_alert = ""
            if game.get('home_rank') and game.get('away_rank') and home_ml and away_ml:
                home_rank = game.get('home_rank', 99)
                away_rank = game.get('away_rank', 99)
                
                if home_rank and away_rank and home_rank != away_rank:
                    # Lower rank number = better team
                    favorite_rank = min(home_rank, away_rank)
                    underdog_rank = max(home_rank, away_rank)
                    rank_diff = underdog_rank - favorite_rank
                    
                    # Check for significant rank differences
                    if rank_diff >= 10:  # Equivalent to about 3-4 seed difference
                        favorite_team = game['home_team'] if home_rank == favorite_rank else game['away_team']
                        underdog_team = game['away_team'] if home_rank == favorite_rank else game['home_team']
                        
                        # Get moneyline for underdog
                        underdog_ml = game.get('away_moneyline') if underdog_team == game['away_team'] else game.get('home_moneyline')
                        
                        if underdog_ml:
                            # Create matchup data for upset detection
                            matchup = {
                                'higher_seed_team': favorite_team,
                                'lower_seed_team': underdog_team,
                                'higher_seed': favorite_rank,
                                'lower_seed': underdog_rank,
                                'underdog_ml': underdog_ml
                            }
                            
                            # Find upset candidates
                            upsets = find_upset_candidates(
                                [matchup], 
                                min_seed_diff=rank_diff,
                                efficiency_data={team['team']: team for team in efficiency_list},
                                stats_data={team['team']: team for team in stats_list},
                                models=models
                            )
                            
                            if upsets:
                                upset = upsets[0]
                                upset_alert = f"🔥 {upset['underdog']} ({upset['edge']:.1%})"

            table_data.append({
                'Date': date_str,
                'Away Team': f"{game['away_team']} {away_rank}".strip(),
                'Home Team': f"{game['home_team']} {home_rank}".strip(),
                'Moneyline': moneyline_str,
                'Model Prob': pred_model_str,
                'Market Prob': market_prob_str,
                'Edge': edge_str,
                'Upset Alert': upset_alert,
                'Spread': spread_str,
                'Total': total_str
            })

        # Display table
        if table_data:
            df = pd.DataFrame(table_data)
            st.dataframe(df, width='stretch', hide_index=True, height=get_dataframe_height(df))
        else:
            st.warning("No games available for display.")

    with tab2:
        st.header("🎯 Individual Game Analysis")
        st.markdown("*Select a specific game for detailed analysis and betting recommendations*")

        # Game selector with date/time
        game_options = []
        for game in games:
            # Format date from game_date field
            if game.get('game_date'):
                try:
                    game_dt = pd.to_datetime(game['game_date'])
                    if game_dt.hour == 0 and game_dt.minute == 0:
                        date_str = game_dt.strftime('%a, %b %d (Time TBD)')
                    else:
                        date_str = game_dt.strftime('%a, %b %d %I:%M %p')
                except:
                    date_str = "Date TBD"
            else:
                date_str = "Date TBD"
            game_options.append(f"{game['away_team']} @ {game['home_team']} - {date_str}")

        selected_game_idx = st.selectbox(
            "Select a game to analyze:",
            range(len(game_options)),
            format_func=lambda x: game_options[x]
        )

        selected_game = games[selected_game_idx]

        # Enrich selected game with CBBD data
        enriched_game = enrich_espn_game_with_cbbd_data(selected_game, efficiency_list, stats_list, season_used)
        if not enriched_game:
            st.error("Unable to load data for this game. Please try another game.")
            return

        # Get advanced metrics for selected game
        advanced_metrics = None
        if kenpom_df is not None or bart_df is not None:
            home_team = normalize_team_name(selected_game['home_team'])
            away_team = normalize_team_name(selected_game['away_team'])
            advanced_metrics = enrich_with_advanced_metrics(home_team, away_team, kenpom_df, bart_df)

        # Make predictions for selected game
        predictions = make_predictions(enriched_game, models, advanced_metrics)

        # Display the prediction
        render_game_prediction(enriched_game, predictions, efficiency_list, stats_list, models)

        st.info("💡 **Note:** Predictions show what the model would have forecasted before the game.")

        # Model details
        with st.expander("🤖 Model Details"):
            st.markdown("""
            **Features Used:**
            - Efficiency ratings (offensive, defensive, net)
            - Scoring statistics (PPG, opponent PPG)
            - Four factors (eFG%, turnover rate, ORB%, FTR)
            - Pace and tempo data

            **Models Trained:**
            - Spread: XGBoost, Random Forest, Linear Regression
            - Total: XGBoost, Random Forest, Linear Regression
            - Moneyline: XGBoost, Random Forest, Logistic Regression

            **Training Data:** 2022 NCAA Regular Season (2,194 games)
            
            **Value Bet Detection:**
            - Identifies underdogs with higher win probability than odds suggest
            - Uses Kelly Criterion for optimal bet sizing
            - Minimum 5% ROI threshold for value identification
            """)

    with tab3:
        st.header("🎲 Parlay Builder")
        st.markdown("Build custom parlays from upcoming games and calculate combined odds and expected value.")
        
        # Import the parlay function
        from features import build_parlay
        
        st.markdown("### Select Games for Your Parlay")
        
        # Get available games with predictions
        parlay_games = []
        for game in games[:10]:  # Limit to first 10 games for UI simplicity
            enriched_game = enrich_espn_game_with_cbbd_data(game, efficiency_list, stats_list, season_used)
            if enriched_game:
                predictions = make_predictions(enriched_game, models, None)
                if 'moneyline' in predictions and predictions['moneyline']:
                    parlay_games.append({
                        'game': game,
                        'enriched': enriched_game,
                        'predictions': predictions
                    })
        
        if not parlay_games:
            st.warning("No games available for parlay building.")
        else:
            # Multiselect for games
            game_options = [f"{g['game']['away_team']} @ {g['game']['home_team']}" for g in parlay_games]
            selected_indices = st.multiselect(
                "Select games to include in your parlay:",
                range(len(parlay_games)),
                format_func=lambda x: game_options[x],
                max_selections=6  # Reasonable limit for parlays
            )
            
            if selected_indices:
                st.markdown("### Your Parlay Picks")
                
                picks = []
                for idx in selected_indices:
                    game_data = parlay_games[idx]
                    game = game_data['game']
                    predictions = game_data['predictions']
                    
                    # Determine which side to bet based on model probability
                    home_prob = predictions['moneyline']['home_win_prob']
                    away_prob = predictions['moneyline']['away_win_prob']
                    
                    # Get moneyline odds (handle None values)
                    home_ml = game.get('home_moneyline') or 0
                    away_ml = game.get('away_moneyline') or 0
                    
                    if home_prob > away_prob:
                        # Bet home team
                        pick = {
                            'team': game['home_team'],
                            'odds': home_ml,
                            'model_prob': home_prob,
                            'opponent': game['away_team']
                        }
                    else:
                        # Bet away team
                        pick = {
                            'team': game['away_team'],
                            'odds': away_ml,
                            'model_prob': away_prob,
                            'opponent': game['home_team']
                        }
                    
                    picks.append(pick)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**{pick['team']}** vs {pick['opponent']}")
                    with col2:
                        if pick['odds'] and pick['odds'] != 0:
                            st.markdown(f"Moneyline: {pick['odds']:+d}")
                        else:
                            st.markdown("Moneyline: N/A")
                    with col3:
                        st.markdown(f"Model Prob: {pick['model_prob']:.1%}")
                
                if st.button("Calculate Parlay", type="primary"):
                    parlay_result = build_parlay(picks)
                    
                    st.success("Parlay Calculated!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Parlay Odds", f"{parlay_result['parlay_odds']:+.0f}")
                    with col2:
                        st.metric("Decimal Odds", f"{parlay_result['decimal_odds']:.2f}")
                    with col3:
                        st.metric("Combined Probability", f"{parlay_result['combined_prob']:.1%}")
                    with col4:
                        ev_color = "green" if parlay_result['is_positive_ev'] else "red"
                        st.metric("Expected Value", f"{parlay_result['expected_value']:.1%}", 
                                delta="+" if parlay_result['is_positive_ev'] else "-")
                    
                    if parlay_result['is_positive_ev']:
                        st.success("✅ Positive Expected Value - This parlay has a mathematical edge!")
                    else:
                        st.warning("⚠️ Negative Expected Value - This parlay may not be profitable long-term.")
            else:
                st.info("Select games above to build your parlay.")

    with tab4:
        st.header("📈 Historical Against the Spread Trends")
        st.markdown("Analyze team's historical against-the-spread performance over the past 5 seasons.")
        
        # Import the ATS analysis function
        from features import analyze_ats_trends
        
        # Team selector
        team_options = sorted(list(set([game['home_team'] for game in games] + [game['away_team'] for game in games])))
        selected_team = st.selectbox("Select a team to analyze:", team_options)
        
        if selected_team:
            with st.spinner(f"Analyzing {selected_team}'s Against the Spread trends..."):
                try:
                    # Normalize team name to match historical data format
                    normalized_team = normalize_team_name(selected_team)
                    ats_results = analyze_ats_trends(normalized_team, years=5)
                    
                    if ats_results and ats_results.get('overall_ats') and (ats_results['overall_ats']['wins'] + ats_results['overall_ats']['losses'] > 0):
                        st.markdown(f"### {selected_team} Against the Spread Performance (Last 5 Seasons)")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            wins = ats_results['overall_ats']['wins']
                            losses = ats_results['overall_ats']['losses']
                            total = wins + losses
                            pct = wins / total if total > 0 else 0
                            st.metric("Overall Against the Spread", f"{wins}-{losses}", f"{pct:.1%}")
                        
                        with col2:
                            fav_wins = ats_results['as_favorite']['wins']
                            fav_losses = ats_results['as_favorite']['losses']
                            fav_total = fav_wins + fav_losses
                            fav_pct = fav_wins / fav_total if fav_total > 0 else 0
                            st.metric("As Favorite", f"{fav_wins}-{fav_losses}", f"{fav_pct:.1%}")
                        
                        with col3:
                            dog_wins = ats_results['as_underdog']['wins']
                            dog_losses = ats_results['as_underdog']['losses']
                            dog_total = dog_wins + dog_losses
                            dog_pct = dog_wins / dog_total if dog_total > 0 else 0
                            st.metric("As Underdog", f"{dog_wins}-{dog_losses}", f"{dog_pct:.1%}")
                        
                        # Additional insights
                        st.markdown("### Key Insights")
                        
                        if pct > 0.52:
                            st.success(f"✅ {selected_team} covers the spread {pct:.1%} of the time - strong Against the Spread performance!")
                        elif pct < 0.48:
                            st.warning(f"⚠️ {selected_team} only covers {pct:.1%} of the time - poor Against the Spread performance.")
                        else:
                            st.info(f"📊 {selected_team} has neutral Against the Spread performance at {pct:.1%}.")
                        
                        # Favorite vs Underdog analysis
                        if fav_pct > dog_pct + 0.05:
                            st.info(f"💪 Performs much better as a favorite ({fav_pct:.1%} vs {dog_pct:.1%} as underdog)")
                        elif dog_pct > fav_pct + 0.05:
                            st.info(f"🎯 Performs much better as an underdog ({dog_pct:.1%} vs {fav_pct:.1%} as favorite)")
                        
                        st.markdown(f"**Data Source:** {total} games with betting lines from {datetime.now().year-5}-{datetime.now().year-1} seasons")
                    
                    else:
                        st.warning(f"No historical Against the Spread data found for {selected_team}.")
                        st.info("This may be due to limited betting line coverage for this team, or the team may be newer to D1 basketball.")
                        
                except Exception as e:
                    st.error(f"Error analyzing {selected_team}: {str(e)}")
                    st.info("This feature requires historical betting data. Make sure the data collection has been run.")

if __name__ == "__main__":
    main()