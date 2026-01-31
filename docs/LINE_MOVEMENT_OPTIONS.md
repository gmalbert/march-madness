# Line Movement Tracking Options

## Current Situation
✅ You have the infrastructure built:
- `opening_line_database.py` - Stores opening lines
- `odds_api_integration.py` - Fetches current lines from Odds API
- Line movement calculation logic

❌ Missing piece: Reliable source of opening lines

---

## Option 1: Manual Opening Lines (FREE) ⭐ Current Recommendation

**How it works:**
- Manually enter opening lines into the database when games are first posted
- Use Odds API to fetch current lines
- System automatically calculates movement

**Implementation:**
```python
# Already built - just use it!
from opening_line_database import OpeningLineDatabase

db = OpeningLineDatabase()
db.add_opening_line(
    home_team="Duke",
    away_team="North Carolina", 
    game_date="2026-02-08",
    spread=-5.5,
    total=145.5,
    home_ml=-220,
    away_ml=+180,
    source="DraftKings"
)
```

**Pros:**
- ✅ Completely free
- ✅ Already built and ready to use
- ✅ Simple and reliable
- ✅ You control data quality

**Cons:**
- ❌ Manual data entry (5-10 min per day for March Madness)
- ❌ Not fully automated
- ❌ Need to check sportsbooks when lines first post

**Best for:** Your current use case - tracking a manageable number of games during March Madness

---

## Option 2: Automated Opening Line Capture (FREE)

**How it works:**
- Run a scheduled script that fetches lines from Odds API every few hours
- When a new game appears with odds, save those as "opening lines"
- Continue tracking for line movement

**Implementation:**
```python
# Create: auto_capture_opening_lines.py
import schedule
import time
from opening_line_database import OpeningLineDatabase
from odds_api_integration import OddsAPIClient

def capture_new_openings():
    """Check for new games and save their first odds as opening lines"""
    db = OpeningLineDatabase()
    odds_client = OddsAPIClient()
    
    current_odds = odds_client.get_live_odds()
    
    for game in current_odds:
        # Check if we already have opening line for this game
        existing = db.get_opening_line(game['home_team'], game['away_team'])
        
        if not existing:
            # This is the first time we've seen odds for this game - save as opening
            db.add_opening_line(
                home_team=game['home_team'],
                away_team=game['away_team'],
                game_date=game['game_date'],
                spread=game['spread'],
                total=game['total'],
                home_ml=game['home_ml'],
                away_ml=game['away_ml'],
                source="Auto-captured"
            )
            print(f"Captured opening line: {game['away_team']} @ {game['home_team']}")

# Run every 2 hours
schedule.every(2).hours.do(capture_new_openings)

while True:
    schedule.run_pending()
    time.sleep(60)
```

**Pros:**
- ✅ Free
- ✅ Automated
- ✅ Uses existing Odds API subscription
- ✅ Captures most opening lines

**Cons:**
- ❌ Might miss true opening if lines post between checks
- ❌ Requires running continuously (on PC or server)
- ❌ Odds API has rate limits (500 requests/month on free tier)

**Best for:** If you want automation and can run a script 24/7

---

## Option 3: Alternative Odds APIs

### A) **The Odds API** (Current - FREE tier available)
- What you're already using
- 500 requests/month free
- Current lines only, no historical

### B) **OddsJam API** (FREE tier)
- https://oddsjam.com/api
- Free tier: 100 requests/day
- Real-time odds from multiple books
- No historical odds in free tier

### C) **RapidAPI Odds Providers**
- Multiple odds APIs available
- Most have small free tiers
- Examples:
  - Odds API (different from The Odds API)
  - LiveScore Odds
  - BetConstruct

### D) **Action Network** (Research needed)
- Known for betting data
- May have API or data access
- Need to check pricing/availability

**Implementation effort:** Medium (need to integrate new API)

**Pros:**
- ✅ Might find better features
- ✅ Could have lower costs
- ✅ Diversify data sources

**Cons:**
- ❌ Still won't have historical opening lines
- ❌ Same problem: need to capture openings in real-time
- ❌ Time to research and integrate

---

## Option 4: Web Scraping (FREE but risky)

**How it works:**
- Scrape odds directly from sportsbook websites
- Store opening lines when first detected
- Continue monitoring for movement

**Target sites:**
- DraftKings Sportsbook
- FanDuel Sportsbook  
- BetMGM
- Caesars

**Implementation:**
```python
# Example with BeautifulSoup/Selenium
from selenium import webdriver
from bs4 import BeautifulSoup

def scrape_draftkings_cbb_odds():
    driver = webdriver.Chrome()
    driver.get("https://sportsbook.draftkings.com/leagues/basketball/ncaab")
    # Parse HTML and extract odds
    # ...
```

**Pros:**
- ✅ Free
- ✅ Access to any sportsbook's data
- ✅ Real-time updates

**Cons:**
- ❌ Violates most sportsbooks' Terms of Service
- ❌ Could get IP banned
- ❌ Fragile - breaks when sites change
- ❌ Requires constant maintenance
- ❌ Legal gray area
- ❌ Requires running headless browsers

**Best for:** Not recommended unless you have no other option

---

## Option 5: Paid Data Services

### A) **SportsDataIO with Replay API** ($50-150/month)
- Historical odds data
- Line movement tracking built-in
- Complete historical seasons for backtesting
- Professional-grade data

**What you'd get:**
- Opening lines for any past game
- Full line movement history
- Tournament data
- Player/team stats

**Cost:** $50-150/month depending on plan

### B) **Don Best Sports** (Enterprise - $$$)
- Industry standard for odds data
- Real-time line movement
- Historical odds database
- Very expensive (thousands/month)

### C) **SportsRadar** (Enterprise - $$$)
- Comprehensive sports data
- Betting odds included
- Very expensive

### D) **BetQL** ($20-50/month)
- Betting analytics platform
- Line movement tracking
- May have API access

**Best for:** If you're serious about this as a business or professional betting

---

## Option 6: Hybrid Approach (Recommended) ⭐

**Combine the best of multiple options:**

1. **During off-hours:** Manual entry for key games
2. **During busy periods:** Automated capture script  
3. **For backtesting:** Use free historical data sources or one-time purchase

**Example workflow:**
```
Morning: Check for new games posted overnight → Manual entry (5 min)
Afternoon: Automated script checks for new lines
Evening: Manual verification of important games
Game day: Real-time line movement via Odds API
```

**Pros:**
- ✅ Free
- ✅ Reliable for important games
- ✅ Automated for convenience
- ✅ Flexible approach

**Cons:**
- ❌ Still requires some manual work
- ❌ Not fully automated

---

## Recommendation Matrix

### For March Madness 2026 (Your immediate need):

| Priority | Option | Why |
|----------|--------|-----|
| 🥇 **#1** | **Manual Entry + Odds API** | Free, reliable, manageable volume |
| 🥈 **#2** | **Automated Capture Script** | Backup for busy tournament days |
| 🥉 **#3** | **Monitor 1-2 sportsbooks directly** | Cross-reference for accuracy |

### For Future Seasons / Scaling:

| If you want... | Choose... |
|----------------|-----------|
| Professional-grade data | SportsDataIO ($50-150/mo) |
| Historical backtesting | SportsDataIO Replay API |
| Fully automated tracking | Automated capture + cloud hosting |
| Budget solution | Manual entry + database |

---

## Quick Start: Manual Entry Workflow

Here's how to use your existing system effectively:

### 1. Daily Routine (5-10 minutes)
```bash
# Morning: Check when lines post for the day's games
# Visit: DraftKings or your preferred sportsbook
# Copy opening lines to spreadsheet or directly to database
```

### 2. Enter into database:
```python
python -c "
from opening_line_database import OpeningLineDatabase
db = OpeningLineDatabase()

# Example: Enter opening lines
db.add_opening_line(
    home_team='Duke',
    away_team='UNC',
    game_date='2026-03-15',
    spread=-6.5,
    total=145.5,
    home_ml=-280,
    away_ml=+220,
    source='DraftKings'
)
print('✅ Opening line added')
"
```

### 3. Track movement:
```python
python odds_api_integration.py
# Automatically compares current lines to opening lines
# Displays movement for all games
```

### 4. During March Madness:
- Lines typically post 24-48 hours before games
- Check sportsbooks once daily
- 5 minutes to enter 10-15 games
- System handles everything else automatically

---

## Bottom Line

**For your current needs:** Stick with **Manual Entry (Option 1)** supplemented by **Automated Capture (Option 2)** for busy days.

**Why:** 
- Free
- Already built
- Reliable
- Perfect for March Madness volume (~60 games over 3 weeks)
- You can always upgrade later if needed

**Total time commitment:** 5-10 minutes per day during tournament season

The infrastructure you built is solid. You don't need expensive APIs to make it work!
