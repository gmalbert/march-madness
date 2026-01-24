#!/usr/bin/env python3
"""
Check if there are college basketball games scheduled for today or tomorrow.
Used by GitHub Actions to determine if efficiency data should be updated.
"""
import requests
from datetime import datetime, timedelta
import sys

def check_for_games():
    """Check if there are games scheduled today or tomorrow."""
    # Check today and tomorrow
    dates_to_check = []
    current_date = datetime.now()

    for i in range(2):  # Today and tomorrow
        check_date = current_date + timedelta(days=i)
        dates_to_check.append(check_date.strftime('%Y%m%d'))

    total_games = 0

    for date_str in dates_to_check:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {"dates": date_str, "limit": 300}

        try:
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                events = data.get('events', [])
                game_count = len(events)
                total_games += game_count
                print(f"Games on {date_str}: {game_count}")
            else:
                print(f"Error checking {date_str}: HTTP {response.status_code}")
        except Exception as e:
            print(f"Error checking {date_str}: {e}")
            continue

    print(f"Total games found: {total_games}")
    return total_games > 0

def is_basketball_season():
    """Check if current date is within college basketball season."""
    now = datetime.now()
    month = now.month
    day = now.day

    # College basketball season typically runs from November to March/April
    # Conservative estimate: October 15 to April 15
    if month == 10 and day >= 15:
        return True
    elif month in [11, 12, 1, 2, 3]:
        return True
    elif month == 4 and day <= 15:
        return True
    else:
        return False

if __name__ == "__main__":
    print("Checking basketball season and scheduled games...")

    if not is_basketball_season():
        print("Not basketball season - skipping efficiency data update")
        sys.exit(1)

    if not check_for_games():
        print("No games scheduled - skipping efficiency data update")
        sys.exit(1)

    print("Basketball season with games scheduled - proceeding with efficiency data update")
    sys.exit(0)