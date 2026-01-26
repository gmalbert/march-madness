# Automated Opening Line Capture - Quick Start Guide

## What This Does

Automatically monitors The Odds API and saves opening lines when new games appear.
- ✅ Runs 24/7 in the background
- ✅ Checks for new games every 2 hours
- ✅ Extra checks during peak posting times (9 AM, 12 PM, 3 PM, 6 PM, 9 PM)
- ✅ Saves everything to your existing opening line database
- ✅ Logs all activity for monitoring

## How to Use

### Option 1: Quick Start (Windows)

Just double-click:
```
start_opening_line_capture.bat
```

The script will:
1. Open a terminal window
2. Start monitoring for new lines
3. Run continuously in the background

**To stop:** Close the terminal window or press Ctrl+C

### Option 2: Manual Start

```bash
# Activate virtual environment
venv\Scripts\activate

# Run the capture script
python auto_capture_opening_lines.py
```

## What You'll See

```
======================================================================
🚀 Opening Line Capture System Started
======================================================================
Database: data_files/opening_lines.json
Check interval: Every 2 hours
API requests remaining: 500
======================================================================

📡 Fetching odds from API... (Request #1)
   API Usage: 1 used, 499 remaining
   Retrieved 47 games with odds

✅ NEW OPENING LINE: Kentucky @ Duke
   Date: 2026-02-15
   Spread: -5.5, Total: 145.5
   ML: +180 / -220
   Source: DraftKings

📊 Capture Summary:
   ✅ New opening lines captured: 12
   ⏭️  Existing lines skipped: 35
   📡 API requests remaining: 499
```

## Check the Log

All activity is saved to:
```
data_files/opening_line_capture.log
```

View it anytime to see:
- When new lines were captured
- What games were found
- API usage stats
- Any errors

## API Rate Limits

**Free Tier:** 500 requests per month

**Current usage:**
- 1 request per check
- Checks every 2 hours = 12 requests/day = ~360 requests/month
- Extra checks at peak times = +5 requests/day = +150 requests/month
- **Total: ~510 requests/month**

**⚠️ This slightly exceeds free tier!**

**Solutions:**
1. Reduce check frequency to every 3 hours (edit the script)
2. Remove some peak-time checks
3. Upgrade to paid tier ($10/month for 10,000 requests)

### Adjust Check Frequency

Edit `auto_capture_opening_lines.py` line 215:
```python
# Change from every 2 hours to every 3 hours
schedule.every(3).hours.do(self.capture_new_opening_lines)
```

Or remove some peak checks (lines 219-223) if you want to stay under 500/month.

## Integration with Your Site

The captured opening lines automatically integrate with:

### 1. Line Movement Tracking
```python
from odds_api_integration import track_line_movement

# Automatically compares current lines to captured opening lines
movement = track_line_movement()
```

### 2. Predictions Display
```python
# Your predictions already pull opening lines from the database
# They'll now be auto-populated!
```

### 3. Manual Viewing
```python
from opening_line_database import OpeningLineDatabase

db = OpeningLineDatabase()
lines = db.get_all_lines()
for line in lines:
    print(f"{line['away_team']} @ {line['home_team']}: {line['spread']}")
```

## Tips

### Run on Startup (Windows)

1. Press `Win + R`
2. Type `shell:startup` and press Enter
3. Create a shortcut to `start_opening_line_capture.bat` in that folder
4. The script will auto-start when you log in

### Run as Background Service

For more advanced setup (always running, even when not logged in):
```bash
# Install NSSM (Non-Sucking Service Manager)
# Download from: https://nssm.cc/download

# Then run:
nssm install OpeningLineCapture "C:\Users\gmalb\Downloads\march-madness\venv\Scripts\python.exe" "C:\Users\gmalb\Downloads\march-madness\auto_capture_opening_lines.py"
```

### Monitor Remotely

Check the log file from anywhere:
```bash
# View last 20 lines
tail -20 data_files/opening_line_capture.log

# Or on Windows PowerShell:
Get-Content data_files/opening_line_capture.log -Tail 20
```

## Troubleshooting

### "ODDS_API_KEY not found"
- Make sure `.env` file exists with `ODDS_API_KEY=your_key_here`
- Restart the script after adding the key

### "API request failed"
- Check your internet connection
- Verify API key is valid at https://the-odds-api.com/
- Check if you've exceeded rate limits

### "No new lines captured"
- Normal if games already have opening lines saved
- Lines typically post 24-48 hours before games
- Check log to see what was skipped

### Script stops unexpectedly
- Check `opening_line_capture.log` for error messages
- Ensure Python virtual environment is activated
- Verify all dependencies are installed: `pip install -r requirements.txt`

## What's Next?

Once the script is running:
1. It captures opening lines automatically
2. Your `odds_api_integration.py` fetches current lines
3. Line movement is calculated automatically
4. Everything shows up in your predictions UI

**You're done!** The system is now fully automated.
