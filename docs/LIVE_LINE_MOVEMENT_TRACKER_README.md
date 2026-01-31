# Live Line Movement Tracker - Complete Implementation

This document provides a complete guide to implementing and using the Live Line Movement Tracker for March Madness betting analysis.

## Overview

The Live Line Movement Tracker enables real-time analysis of betting line movements in NCAA Basketball games. It combines:

- **Live Odds**: Current betting lines from The Odds API (free tier)
- **Opening Lines**: Historical opening lines stored in a local database
- **Line Movement Analysis**: Calculation of how lines have moved and identification of sharp money

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Odds API      │    │ Opening Line DB  │    │ Line Movement   │
│   (Live Data)   │───▶│  (Historical)    │───▶│   Analysis      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
       ▲                       ▲                       │
       │                       │                       ▼
       └───────────────────────┴─────────────────▶ Features.py
```

## Components

### 1. Odds API Integration (`odds_api_integration.py`)
- Fetches live betting odds from The Odds API
- Caches odds data locally
- Provides current line extraction

### 2. Opening Line Database (`opening_line_database.py`)
- Stores historical opening lines
- Supports multiple data sources
- Provides fallback estimation methods

### 3. SportsDataIO Integration (`sportsdataio_integration.py`)
- Populates database with real historical data
- Requires free trial signup at SportsDataIO

### 4. Features Integration (`features.py`)
- Main interface for line movement tracking
- Combines live and historical data

## Setup Instructions

### Step 1: Get API Keys

#### The Odds API (Free)
1. Sign up at [https://the-odds-api.com/](https://the-odds-api.com/)
2. Get your free API key (500 requests/month)
3. Add to `.env` file: `ODDS_API_KEY=your_key_here`

#### SportsDataIO (Free Trial)
1. Sign up at [https://sportsdata.io/](https://sportsdata.io/)
2. Get free trial API key
3. Add to `.env` file: `SPORTS_DATA_IO_KEY=your_key_here`

### Step 2: Install Dependencies

```bash
pip install requests python-dotenv
```

### Step 3: Populate Opening Lines Database

#### Option A: Use SportsDataIO (Recommended)
```bash
python sportsdataio_integration.py
```
Choose option 1 or 2 to populate with real historical data.

#### Option B: Manual/Sample Data
```python
from opening_line_database import OpeningLineDatabase
db = OpeningLineDatabase()
# Add sample games manually or import from CSV
```

## Usage Examples

### Basic Line Movement Tracking

```python
from odds_api_integration import track_line_movement

# Track line movement for a game
result = track_line_movement(("Duke", "North Carolina"))
print(result)
```

### Complete Analysis Workflow

```python
# 1. Cache fresh odds
from odds_api_integration import cache_odds_snapshot
cache_odds_snapshot()

# 2. Run line movement analysis
from odds_api_integration import demo_line_movement_tracking
demo_line_movement_tracking()
```

### Database Management

```python
from opening_line_database import OpeningLineDatabase

db = OpeningLineDatabase()

# Add opening line
db.add_opening_line(
    game_id="2024_001",
    home_team="Duke",
    away_team="North Carolina",
    opening_spread=-3.5,
    opening_total=145.5,
    game_date="2024-01-15",
    season="2023-24"
)

# Get season statistics
stats = db.get_season_stats("2023-24")
print(f"Average spread: {stats['avg_spread']}")

# Export to CSV
db.export_to_csv("opening_lines.csv")
```

## Data Sources

### Free Options Available

1. **SportsDataIO** (Recommended)
   - Historical odds from 2019+
   - College basketball focus
   - Free trial available
   - Opening lines and line movement data

2. **The Odds API**
   - Live odds only
   - Free tier: 500 requests/month
   - Current lines, no historical data

3. **Kaggle Datasets**
   - Limited college basketball betting data
   - Mostly soccer/football datasets
   - No comprehensive NCAA data found

### Data Limitations

- **Free APIs**: Generally provide live data only
- **Historical Data**: Requires paid plans or manual collection
- **Opening Lines**: Not available on free Odds API plans
- **Coverage**: Most free sources focus on professional sports

## Analysis Features

### Line Movement Calculation

The system calculates:
- **Spread Movement**: Change from opening to current spread
- **Total Movement**: Change in over/under line
- **Sharp Money Indicators**: Identifies heavy betting on one side

### Movement Thresholds

- **3+ points**: Heavy money movement
- **1-3 points**: Moderate movement
- **<1 point**: Minimal movement

### Fallback Methods

When opening lines aren't available:
1. Use similar historical games
2. Estimate based on team strength
3. Use current lines as proxy (with disclaimer)

## Integration with Existing Code

The line movement tracker integrates with your existing `features.py`:

```python
# In features.py, you can now call:
from odds_api_integration import get_current_lines, get_opening_lines

current = get_current_lines(("Duke", "UNC"))
opening = get_opening_lines(("Duke", "UNC"))

if current and opening:
    movement = calculate_line_movement(current, opening)
    # Use movement data in your analysis
```

## Testing

### Run the Demo

```bash
python odds_api_integration.py
```

This will:
- Populate sample opening lines
- Fetch live odds
- Show line movement analysis
- Display database statistics

### Test SportsDataIO Integration

```bash
python sportsdataio_integration.py
```

Follow the prompts to test API connection and populate real data.

## Troubleshooting

### Common Issues

1. **No API Key**: Check `.env` file and environment variables
2. **Rate Limits**: Free APIs have request limits
3. **Team Name Mismatches**: Normalize team names for matching
4. **No Historical Data**: Use fallback methods or manual data entry

### Debug Commands

```python
# Check API connection
from odds_api_integration import odds_client
print(odds_client.get_ncaab_odds())

# Check database contents
from opening_line_database import OpeningLineDatabase
db = OpeningLineDatabase()
stats = db.get_season_stats("2023-24")
print(stats)
```

## Future Enhancements

### Potential Improvements

1. **Automated Data Collection**: Scheduled jobs to update opening lines
2. **Machine Learning**: Predict line movements based on historical patterns
3. **Multiple Bookmakers**: Compare lines across different sportsbooks
4. **Real-time Alerts**: Notifications when lines move significantly
5. **Advanced Analytics**: Incorporate weather, injuries, and other factors

### Additional Data Sources

- **Action Network**: Sports betting news and analysis
- **Pinnacle Sports**: Historical odds archive
- **BetExplorer**: Historical betting data
- **OddsPortal**: Historical odds comparison

## Conclusion

This implementation provides a complete foundation for live line movement tracking in March Madness. While free data sources have limitations, the modular design allows for easy integration of additional paid data sources as needed.

The system successfully combines live odds with historical opening lines to provide valuable betting analysis insights, with graceful fallbacks when complete data isn't available.