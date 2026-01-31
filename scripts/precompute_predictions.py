"""
Pre-compute game predictions and save to JSON for fast loading in Streamlit.

This script fetches ESPN games, enriches them with team data, generates predictions,
and saves the results to JSON files. This eliminates the need to make API calls
every time the predictions page loads.

Usage:
    python scripts/precompute_predictions.py
    python scripts/precompute_predictions.py --date 2025-03-15
"""

import sys
import os
from pathlib import Path
from datetime import datetime
import json
import argparse
import pandas as pd
import numpy as np
import joblib
from typing import Dict, List, Optional
import warnings

# Suppress version compatibility warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*version.*')
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data_collection import fetch_games, fetch_betting_lines, fetch_adjusted_efficiency, fetch_team_stats
from data_tools.efficiency_loader import EfficiencyDataLoader
from fetch_live_odds import fetch_live_odds

# Import prediction functions
import importlib.util
spec = importlib.util.spec_from_file_location("predictions", Path(__file__).parent.parent / "predictions.py")
predictions_module = importlib.util.module_from_spec(spec)
sys.modules["predictions"] = predictions_module

# Configuration
DATA_DIR = Path("data_files")
MODEL_DIR = DATA_DIR / "models"
OUTPUT_DIR = DATA_DIR / "precomputed_predictions"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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
        try:
            models['spread_scalers'] = {
                'linear': joblib.load(MODEL_DIR / 'spread_linear_regression_scaler.joblib')
            }
        except:
            models['spread_scalers'] = {}
        print("[OK] Loaded spread models")
    except Exception as e:
        print(f"[WARNING] Could not load spread models: {e}")
        models['spread'] = None

    # Total models
    try:
        models['total'] = {
            'xgboost': joblib.load(MODEL_DIR / 'total_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'total_random_forest.joblib'),
            'linear': joblib.load(MODEL_DIR / 'total_linear_regression.joblib')
        }
        try:
            models['total_scalers'] = {
                'linear': joblib.load(MODEL_DIR / 'total_linear_regression_scaler.joblib')
            }
        except:
            models['total_scalers'] = {}
        print("[OK] Loaded total models")
    except Exception as e:
        print(f"[WARNING] Could not load total models: {e}")
        models['total'] = None

    # Moneyline models
    try:
        models['moneyline'] = {
            'xgboost': joblib.load(MODEL_DIR / 'moneyline_xgboost.joblib'),
            'random_forest': joblib.load(MODEL_DIR / 'moneyline_random_forest.joblib'),
            'logistic': joblib.load(MODEL_DIR / 'moneyline_logistic_regression.joblib')
        }
        try:
            models['moneyline_scalers'] = {
                'logistic': joblib.load(MODEL_DIR / 'moneyline_logistic_regression_scaler.joblib')
            }
        except:
            models['moneyline_scalers'] = {}
        print("[OK] Loaded moneyline models")
    except Exception as e:
        print(f"[WARNING] Could not load moneyline models: {e}")
        models['moneyline'] = None

    return models


def load_espn_games() -> pd.DataFrame:
    """Load ESPN game data from CSV."""
    espn_file = DATA_DIR / "espn_cbb_current_season.csv"
    if espn_file.exists():
        return pd.read_csv(espn_file)
    return pd.DataFrame()


def get_team_data(year: int = 2025):
    """Get team efficiency and stats data for a given year."""
    try:
        # Try the specified year first
        efficiency_list = fetch_adjusted_efficiency(year)
        stats_list = fetch_team_stats(year)
        
        if efficiency_list and stats_list:
            return efficiency_list, stats_list, year
        
        # Fall back to previous year if current year not available
        print(f"[WARNING] {year} data not available, trying {year - 1}...")
        efficiency_list = fetch_adjusted_efficiency(year - 1)
        stats_list = fetch_team_stats(year - 1)
        
        return efficiency_list, stats_list, year - 1
        
    except Exception as e:
        print(f"[ERROR] Could not fetch team data: {e}")
        return None, None, None


def get_kenpom_barttorvik_data():
    """Load KenPom and BartTorvik advanced metrics."""
    try:
        loader = EfficiencyDataLoader()
        
        # Load KenPom
        kenpom_df = None
        try:
            kenpom_df = loader.load_kenpom()
            if kenpom_df is not None and not kenpom_df.empty:
                print(f"[OK] Loaded KenPom data ({len(kenpom_df)} teams)")
        except Exception as e:
            print(f"[WARNING] Could not load KenPom data: {e}")
        
        # Load BartTorvik
        bart_df = None
        try:
            bart_df = loader.load_barttorvik()
            if bart_df is not None and not bart_df.empty:
                print(f"[OK] Loaded BartTorvik data ({len(bart_df)} teams)")
        except Exception as e:
            print(f"[WARNING] Could not load BartTorvik data: {e}")
        
        return kenpom_df, bart_df
        
    except Exception as e:
        print(f"[ERROR] Error loading advanced metrics: {e}")
        return None, None


def normalize_team_name(espn_name: str) -> str:
    """Convert ESPN team name to standard format by removing mascots."""
    
    # Special cases that need exact mapping
    special_cases = {
        'Saint Francis Red Flash': 'St. Francis (PA)',
        'Saint Francis': 'St. Francis (PA)',
    }
    
    if espn_name in special_cases:
        return special_cases[espn_name]
    
    # Common patterns: remove mascots/nicknames
    mascots = [
        # Multi-word mascots - MUST come FIRST before single-word variants
        'Tar Heels', 'Blue Devils', 'Fighting Irish', 'Golden Flashes', 'Red Raiders',
        'Golden Knights', 'Thundering Herd', 'Crimson Tide', 'Mean Green', 'Fighting Illini',
        'Demon Deacons', 'Golden Gophers', 'Yellow Jackets', 'Red Flash', 'Purple Aces', 
        'Scarlet Knights', 'Nittany Lions', 'Red Storm', 'Sun Devils', 'Bluejays',
        'Horned Frogs', 'Blue Demons', 'Blue Hens', 'Rainbow Warriors',
        'Running Bulldogs', 'Green Wave', 'Chanticleers', 'Golden Bears', 
        'Golden Grizzlies', 'Golden Eagles', 'Blue Raiders', 'River Hawks',
        'Black Bears', 'Red Foxes', 'Ragin\' Cajuns', 'Screaming Eagles',
        'Blue Hose', 'Runnin\' Bulldogs',
        # Single-word mascots
        'Wolverines', 'Hoosiers', 'Cyclones', 'Knights', 'Gators', 'Tigers',
        'Wolfpack', 'Dukes', 'Billikens', 'Bonnies', 'Buckeyes', 'Flashes', 
        'RedHawks', 'Ducks', 'Spartans', 'Bears', 'Raiders', 'Razorbacks', 
        'Commodores', 'Bulldogs', 'Bruins', 'Boilermakers', 'Buffaloes', 'Jayhawks', 
        'Wildcats', 'Aggies', 'Huskies', 'Cardinals', 'Sooners', 'Longhorns',
        'Volunteers', 'Gamecocks', 'Rebels', 'Broncos', 'Cougars', 'Panthers', 
        'Eagles', 'Owls', 'Rams', 'Bulls', 'Bearcats', 'Terrapins', 'Cornhuskers', 
        'Waves', 'Cavaliers', 'Mountaineers', 'Hokies', 'Cowboys', 'Utes', 'Dons', 
        'Dolphins', 'Chargers', 'Skyhawks', 'Lakers', 'Mastodons', 'Jaguars', 
        'Seahawks', 'Sharks', 'Salukis', 'Trojans', 'Badgers', 'Friars', 
        'Minutemen', 'Flyers', 'Braves', 'Cardinal', 'Flames', 'Gaels', 
        'Grizzlies', 'Jaspers', 'Kangaroos', 'Leopards', 'Mavericks', 'Pilots', 
        'Redhawks', 'Stags', 'Mustangs', 'Explorers', 'Musketeers', 'Orange', 
        'Beavers', 'Titans', 'Lancers', 'Vikings', 'Dragons', 'Phoenix', 
        'Bengals', 'Vandals', 'Redbirds', 'Seawolves', 'Retrievers',
        'Catamounts', 'Penguins', 'Buccaneers', 'Pride', 'Seminoles',
        'Demon', 'Demons', 'Beach', 'Roos', 'Lions', 'Hurricanes', 'Hawks',
        'Racers', 'Highlanders', 'Lumberjacks', 'Norse', 'Bobcats', 'Colonials',
        'Hornets', 'Sycamores', 'Jackrabbits', 'Thunderbirds', 'Tommies', 'Anteaters',
        'Gauchos', 'Trailblazers', 'Beacons', 'Leathernecks', 'Tribe', 'Warriors',
        'Ramblers', 'Foxes', 'Roadrunners', 'Scarlet', 'Golden', 'Blue',
        'Screaming', 'Red', 'Black', 'Pioneers', 'Tritons'
    ]
    
    # Remove mascot from end of name
    name = espn_name
    for mascot in mascots:
        if name.endswith(f' {mascot}'):
            name = name[:-len(mascot)-1]
            break
    
    return name


def enrich_with_advanced_metrics(home_team: str, away_team: str, kenpom_df: pd.DataFrame, bart_df: pd.DataFrame):
    """Enrich game with advanced metrics from KenPom and BartTorvik."""
    advanced_metrics = {}
    
    # Get KenPom metrics
    if kenpom_df is not None:
        # Determine the team name column
        team_col = 'TeamName' if 'TeamName' in kenpom_df.columns else 'Team'
        
        home_kenpom = kenpom_df[kenpom_df[team_col] == home_team]
        away_kenpom = kenpom_df[kenpom_df[team_col] == away_team]
        
        if not home_kenpom.empty:
            advanced_metrics['home_kenpom'] = home_kenpom.iloc[0].to_dict()
        if not away_kenpom.empty:
            advanced_metrics['away_kenpom'] = away_kenpom.iloc[0].to_dict()
    
    # Get BartTorvik metrics
    if bart_df is not None:
        # Determine the team name column
        team_col = 'TeamName' if 'TeamName' in bart_df.columns else 'Team'
        
        home_bart = bart_df[bart_df[team_col] == home_team]
        away_bart = bart_df[bart_df[team_col] == away_team]
        
        if not home_bart.empty:
            advanced_metrics['home_barttorvik'] = home_bart.iloc[0].to_dict()
        if not away_bart.empty:
            advanced_metrics['away_barttorvik'] = away_bart.iloc[0].to_dict()
    
    return advanced_metrics if advanced_metrics else None


def enrich_espn_game_with_cbbd_data(game_row, efficiency_list, stats_list, season_used):
    """Enrich ESPN game with CBBD efficiency and stats data."""
    try:
        # Get team names
        home_team_espn = game_row.get('home_team')
        away_team_espn = game_row.get('away_team')
        
        if not home_team_espn or not away_team_espn:
            return None
        
        # Normalize team names for matching
        home_team_normalized = normalize_team_name(home_team_espn)
        away_team_normalized = normalize_team_name(away_team_espn)
        
        # Find efficiency data
        home_eff = next((e for e in efficiency_list if normalize_team_name(e.get('team', '')) == home_team_normalized), None)
        away_eff = next((e for e in efficiency_list if normalize_team_name(e.get('team', '')) == away_team_normalized), None)
        
        # Find stats data
        home_stats = next((s for s in stats_list if normalize_team_name(s.get('team', '')) == home_team_normalized), None)
        away_stats = next((s for s in stats_list if normalize_team_name(s.get('team', '')) == away_team_normalized), None)
        
        # Skip if we don't have data for both teams
        if not home_eff or not away_eff or not home_stats or not away_stats:
            return None
        
        # Convert to dicts
        home_eff_dict = {
            'offensiveRating': float(home_eff.get('offensive_rating', 100)),
            'defensiveRating': float(home_eff.get('defensive_rating', 100)),
            'netRating': float(home_eff.get('net_rating', 0))
        }
        
        away_eff_dict = {
            'offensiveRating': float(away_eff.get('offensive_rating', 100)),
            'defensiveRating': float(away_eff.get('defensive_rating', 100)),
            'netRating': float(away_eff.get('net_rating', 0))
        }
        
        home_stats_dict = {
            'ppg': float(home_stats.get('points_per_game', 70)),
            'pace': float(home_stats.get('pace', 68)),
            'efg_pct': float(home_stats.get('effective_fg_pct', 0.50)),
            'to_rate': float(home_stats.get('turnover_pct', 0.15)),
            'orb_pct': float(home_stats.get('orb_pct', 0.30)),
            'ft_rate': float(home_stats.get('ft_rate', 0.35)),
            'opp_ppg': float(home_stats.get('opp_points_per_game', 70))
        }
        
        away_stats_dict = {
            'ppg': float(away_stats.get('points_per_game', 70)),
            'pace': float(away_stats.get('pace', 68)),
            'efg_pct': float(away_stats.get('effective_fg_pct', 0.50)),
            'to_rate': float(away_stats.get('turnover_pct', 0.15)),
            'orb_pct': float(away_stats.get('orb_pct', 0.30)),
            'ft_rate': float(away_stats.get('ft_rate', 0.35)),
            'opp_ppg': float(away_stats.get('opp_points_per_game', 70))
        }
        
        # Get betting lines - handle both dict and Series
        if isinstance(game_row, dict):
            betting_spread = game_row.get('spread')
            betting_over_under = game_row.get('over_under')
            home_ml = game_row.get('home_moneyline')
            away_ml = game_row.get('away_moneyline')
        else:
            betting_spread = game_row.get('spread', None)
            betting_over_under = game_row.get('over_under', None)
            home_ml = game_row.get('home_moneyline', None)
            away_ml = game_row.get('away_moneyline', None)
        
        return {
            'home_team': home_team_espn,
            'away_team': away_team_espn,
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
            'game_date': game_row.get('date', '') if isinstance(game_row, dict) else game_row.get('date'),
            'status': game_row.get('status', '') if isinstance(game_row, dict) else game_row.get('status'),
            'venue': game_row.get('venue', '') if isinstance(game_row, dict) else game_row.get('venue'),
            'neutral_site': game_row.get('neutral_site', False) if isinstance(game_row, dict) else game_row.get('neutral_site', False),
            'home_rank': game_row.get('home_rank') if (isinstance(game_row, dict) and game_row.get('home_rank') != 99) else (game_row.get('home_rank', None) if game_row.get('home_rank', 99) != 99 else None),
            'away_rank': game_row.get('away_rank') if (isinstance(game_row, dict) and game_row.get('away_rank') != 99) else (game_row.get('away_rank', None) if game_row.get('away_rank', 99) != 99 else None),
            'season_used': season_used
        }
    except Exception as e:
        print(f"[ERROR] Could not enrich game: {e}")
        return None


def make_predictions(game: Dict, models: Dict, advanced_metrics: Optional[Dict] = None) -> Dict:
    """Generate predictions for a game."""
    # Predictions will be generated when loading in the Streamlit app
    # We're just pre-computing the enriched game data here
    return {}


def serialize_predictions(predictions: Dict) -> Dict:
    """Convert predictions to JSON-serializable format."""
    serialized = {}
    for key, value in predictions.items():
        if isinstance(value, dict):
            serialized[key] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                              for k, v in value.items()}
        else:
            serialized[key] = value
    return serialized


def precompute_predictions(target_date: Optional[str] = None):
    """Pre-compute predictions and save to JSON."""
    print("=" * 80)
    print("PRE-COMPUTING GAME PREDICTIONS")
    print("=" * 80)
    
    # Load models
    print("\nLoading prediction models...")
    models = load_models()
    
    # Load advanced metrics
    print("\nLoading advanced metrics...")
    kenpom_df, bart_df = get_kenpom_barttorvik_data()
    
    # Load team data
    print("\nLoading team data...")
    efficiency_list, stats_list, season_used = get_team_data()
    
    if not efficiency_list or not stats_list:
        print("[ERROR] Could not load team data")
        return
    
    print(f"[OK] Using {season_used} season data")
    
    # Load ESPN games
    print("\nLoading ESPN games...")
    espn_df = load_espn_games()
    
    if espn_df.empty:
        print("[ERROR] No ESPN game data found. Run fetch_espn_cbb_scores.py first.")
        return
    
    print(f"[OK] Loaded {len(espn_df)} games from ESPN")
    
    # Filter for upcoming games
    espn_df['date_dt'] = pd.to_datetime(espn_df['date'])
    upcoming = espn_df[espn_df['date_dt'] > pd.Timestamp.now(tz='UTC')].copy()
    
    if len(upcoming) == 0:
        print("[WARNING] No upcoming games found. Using recent games instead.")
        upcoming = espn_df.sort_values('date_dt', ascending=False).head(20)
    
    print(f"[OK] Processing {len(upcoming)} games")
    
    # Fetch live odds
    print("\nFetching live betting odds...")
    live_odds = {}
    try:
        live_odds = fetch_live_odds()
        if live_odds:
            print(f"[OK] Fetched odds for {len(live_odds)} games")
        else:
            print("[WARNING] No live odds available")
    except Exception as e:
        print(f"[WARNING] Could not fetch live odds: {e}")
    
    # Enrich and predict
    print("\nEnriching games with team data...")
    enriched_games = []
    skipped_count = 0
    
    for idx, game_row in upcoming.iterrows():
        # Enrich with team data
        enriched = enrich_espn_game_with_cbbd_data(game_row, efficiency_list, stats_list, season_used)
        
        if not enriched:
            skipped_count += 1
            continue
        
        # Add live odds if available
        home_team = enriched['home_team']
        away_team = enriched['away_team']
        
        # Try to match odds by team names (normalize both sides)
        odds_key = None
        for key in live_odds.keys():
            # Key format is typically "away_team @ home_team" or similar
            if (normalize_team_name(home_team) in key or normalize_team_name(away_team) in key or
                home_team in key or away_team in key):
                odds_key = key
                break
        
        if odds_key and odds_key in live_odds:
            odds = live_odds[odds_key]
            enriched['betting_spread'] = odds.get('home_spread') or enriched.get('betting_spread')
            enriched['betting_over_under'] = odds.get('total_line') or enriched.get('betting_over_under')
            enriched['home_moneyline'] = odds.get('home_moneyline') or enriched.get('home_moneyline')
            enriched['away_moneyline'] = odds.get('away_moneyline') or enriched.get('away_moneyline')
            enriched['home_ml'] = enriched['home_moneyline']
            enriched['away_ml'] = enriched['away_moneyline']
        
        if not enriched:
            skipped_count += 1
            continue
        
        # Get advanced metrics
        advanced_metrics = None
        if kenpom_df is not None or bart_df is not None:
            home_team = normalize_team_name(enriched['home_team'])
            away_team = normalize_team_name(enriched['away_team'])
            advanced_metrics = enrich_with_advanced_metrics(home_team, away_team, kenpom_df, bart_df)
        
        # Make predictions
        predictions = make_predictions(enriched, models, advanced_metrics)
        
        # Add predictions to game data
        enriched['predictions'] = serialize_predictions(predictions)
        enriched_games.append(enriched)
    
    if skipped_count > 0:
        print(f"[WARNING] Skipped {skipped_count} games due to missing team data")
    
    print(f"[OK] Generated predictions for {len(enriched_games)} games")
    
    # Save to JSON
    output_file = OUTPUT_DIR / f"predictions_{datetime.now().strftime('%Y-%m-%d')}.json"
    
    output_data = {
        'computed_at': datetime.now().isoformat(),
        'season_used': season_used,
        'num_games': len(enriched_games),
        'games': enriched_games
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[OK] Saved predictions to {output_file}")
    print(f"    - {len(enriched_games)} games")
    print(f"    - Season: {season_used}")
    print(f"    - Timestamp: {output_data['computed_at']}")
    print("\n" + "=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Pre-compute game predictions')
    parser.add_argument('--date', type=str, help='Target date (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    try:
        precompute_predictions(args.date)
    except Exception as e:
        print(f"\n[ERROR] Pre-computation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
