#!/usr/bin/env python3
"""
Automated Opening Line Capture

This script continuously monitors the Odds API for new games and automatically
saves their first appearance as opening lines in the database.

Usage:
    python auto_capture_opening_lines.py

The script will:
1. Check Odds API every 150 minutes for new games
2. When a new game appears, save its odds as opening lines
3. Continue running 24/7 to catch all new lines as they post
4. Log all activity for monitoring
"""

import os
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
import schedule
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from opening_line_database import OpeningLineDatabase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_files/opening_line_capture.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
ODDS_API_KEY = os.environ.get('ODDS_API_KEY')
ODDS_API_BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "basketball_ncaab"
REGIONS = "us"
MARKETS = "h2h,spreads,totals"
ODDS_FORMAT = "american"

class OpeningLineCapture:
    """Automatically capture opening lines from Odds API."""
    
    def __init__(self):
        self.db = OpeningLineDatabase()
        self.api_key = ODDS_API_KEY
        self.requests_used = 0
        self.requests_remaining = 500  # Free tier limit
        
        if not self.api_key:
            raise ValueError("ODDS_API_KEY not found in environment variables")
        
        logger.info("="*70)
        logger.info("Opening Line Capture System Started")
        logger.info("="*70)
        logger.info(f"Database: {self.db.opening_lines_file}")
        logger.info(f"Check interval: Every 150 minutes")
        logger.info(f"API requests remaining: {self.requests_remaining}")
        logger.info("="*70)
    
    def get_current_odds(self) -> Optional[List[Dict]]:
        """Fetch current odds from The Odds API."""
        url = f"{ODDS_API_BASE_URL}/sports/{SPORT}/odds/"
        params = {
            'apiKey': self.api_key,
            'regions': REGIONS,
            'markets': MARKETS,
            'oddsFormat': ODDS_FORMAT
        }
        
        try:
            logger.info(f"Fetching odds from API... (Request #{self.requests_used + 1})")
            response = requests.get(url, params=params, timeout=30)
            
            # Track API usage from headers
            if 'x-requests-used' in response.headers:
                self.requests_used = int(response.headers['x-requests-used'])
            if 'x-requests-remaining' in response.headers:
                self.requests_remaining = int(response.headers['x-requests-remaining'])
            
            logger.info(f"   API Usage: {self.requests_used} used, {self.requests_remaining} remaining")
            
            response.raise_for_status()
            games = response.json()
            logger.info(f"   Retrieved {len(games)} games with odds")
            return games
            
        except requests.exceptions.RequestException as e:
            logger.error(f" API request failed: {e}")
            return None
    
    def parse_game_odds(self, game_data: Dict) -> Optional[Dict]:
        """Extract relevant odds data from API response."""
        try:
            home_team = game_data.get('home_team', '')
            away_team = game_data.get('away_team', '')
            commence_time = game_data.get('commence_time', '')
            
            # Parse commence time
            try:
                game_date = datetime.fromisoformat(commence_time.replace('Z', '+00:00'))
                game_date_str = game_date.strftime('%Y-%m-%d')
            except:
                game_date_str = datetime.now().strftime('%Y-%m-%d')
            
            # Extract odds from bookmakers
            bookmakers = game_data.get('bookmakers', [])
            if not bookmakers:
                return None
            
            # Use first available bookmaker (usually DraftKings or FanDuel)
            bookmaker = bookmakers[0]
            source = bookmaker.get('title', 'Unknown')
            markets = bookmaker.get('markets', [])
            
            # Initialize odds values
            spread = None
            total = None
            home_ml = None
            away_ml = None
            
            # Extract each market type
            for market in markets:
                market_key = market.get('key')
                outcomes = market.get('outcomes', [])
                
                if market_key == 'spreads' and len(outcomes) >= 2:
                    for outcome in outcomes:
                        if outcome['name'] == home_team:
                            spread = outcome.get('point')
                
                elif market_key == 'totals' and len(outcomes) >= 1:
                    total = outcomes[0].get('point')
                
                elif market_key == 'h2h' and len(outcomes) >= 2:
                    for outcome in outcomes:
                        if outcome['name'] == home_team:
                            home_ml = outcome.get('price')
                        elif outcome['name'] == away_team:
                            away_ml = outcome.get('price')
            
            return {
                'home_team': home_team,
                'away_team': away_team,
                'game_date': game_date_str,
                'spread': spread,
                'total': total,
                'home_ml': home_ml,
                'away_ml': away_ml,
                'source': source,
                'commence_time': commence_time
            }
            
        except Exception as e:
            logger.error(f" Error parsing game data: {e}")
            return None
    
    def capture_new_opening_lines(self):
        """Main function to check for and capture new opening lines."""
        logger.info("\n" + "="*70)
        logger.info(f"Checking for new opening lines - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("="*70)
        
        # Check API rate limits
        if self.requests_remaining < 10:
            logger.warning(f"WARNING: Low API requests remaining: {self.requests_remaining}")
            logger.warning("   Consider reducing check frequency to conserve requests")
        
        # Fetch current odds
        games = self.get_current_odds()
        if not games:
            logger.warning("WARNING: No games data received from API")
            return
        
        new_lines_captured = 0
        existing_lines_skipped = 0
        errors = 0
        
        # Process each game
        for game_data in games:
            odds = self.parse_game_odds(game_data)
            if not odds:
                errors += 1
                continue
            
            # Check if we already have opening line for this game
            # Generate game_id from teams and date
            game_id = f"{odds['away_team']}@{odds['home_team']}_{odds['game_date']}"
            
            existing = self.db.get_opening_line(game_id)
            
            if existing:
                existing_lines_skipped += 1
                continue
            
            # This is a new game - save as opening line!
            try:
                self.db.add_opening_line(
                    game_id=game_id,
                    home_team=odds['home_team'],
                    away_team=odds['away_team'],
                    game_date=odds['game_date'],
                    opening_spread=odds['spread'] if odds['spread'] else 0.0,
                    opening_total=odds['total'] if odds['total'] else 0.0,
                    season="2024-25"
                )
                
                logger.info(f" NEW OPENING LINE: {odds['away_team']} @ {odds['home_team']}")
                logger.info(f"   Date: {odds['game_date']}")
                logger.info(f"   Spread: {odds['spread']}, Total: {odds['total']}")
                logger.info(f"   ML: {odds['away_ml']} / {odds['home_ml']}")
                logger.info(f"   Source: {odds['source']}")
                
                new_lines_captured += 1
                
            except Exception as e:
                logger.error(f" Error saving opening line: {e}")
                errors += 1
        
        # Summary
        logger.info("\n" + "-"*70)
        logger.info(" Capture Summary:")
        logger.info(f"    New opening lines captured: {new_lines_captured}")
        logger.info(f"     Existing lines skipped: {existing_lines_skipped}")
        if errors > 0:
            logger.info(f"    Errors: {errors}")
        logger.info(f"    API requests remaining: {self.requests_remaining}")
        logger.info("-"*70)
        
        if new_lines_captured > 0:
            logger.info(f" Successfully captured {new_lines_captured} new opening lines!")
    
    def run_scheduler(self):
        """Run the scheduled capture process."""
        # Run immediately on startup
        self.capture_new_opening_lines()
        
        # Schedule regular checks every 150 minutes
        schedule.every(150).minutes.do(self.capture_new_opening_lines)
        
        # During peak times (9 AM - 11 PM), check more frequently
        # These are when sportsbooks typically post new lines
        schedule.every().day.at("09:00").do(self.capture_new_opening_lines)
        schedule.every().day.at("12:00").do(self.capture_new_opening_lines)
        schedule.every().day.at("15:00").do(self.capture_new_opening_lines)
        schedule.every().day.at("18:00").do(self.capture_new_opening_lines)
        schedule.every().day.at("21:00").do(self.capture_new_opening_lines)
        
        logger.info("\n Scheduled checks:")
        logger.info("    Every 150 minutes (continuous)")
        logger.info("    Extra checks at 9 AM, 12 PM, 3 PM, 6 PM, 9 PM")
        logger.info("\n  Press Ctrl+C to stop\n")
        
        # Run forever
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute for scheduled tasks
        except KeyboardInterrupt:
            logger.info("\n\n Stopping opening line capture...")
            logger.info(f" Final stats: {self.requests_used} API requests used")
            logger.info(" Goodbye!\n")

def main():
    """Main entry point."""
    try:
        capture = OpeningLineCapture()
        capture.run_scheduler()
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
