"""
Fetch upcoming College Basketball games from ESPN API for March Madness predictions.
Focus on current season games (both played and upcoming) for prediction purposes.
"""
import requests
import pandas as pd
from datetime import datetime, timedelta

all_scores = []
seen_events = set()

# Only fetch current season (2025-2026 season = year 2026)
current_year = 2026
print(f"Fetching {current_year} season data (upcoming and recent games)...")

# Try different season types
# seasontype=2: Regular Season
# seasontype=3: Postseason (Conference Tournaments)
# seasontype=4: Tournament (NCAA Tournament)
for seasontype in (2, 3, 4):
    seasontype_name = {2: "Regular", 3: "Postseason", 4: "Tournament"}[seasontype]
    print(f"  Fetching {seasontype_name} season...")
    
    # ESPN's college basketball API can return all games for a season
    # We'll try with just year and seasontype, then paginate if needed
    limit = 300  # Max games per request
    offset = 0
    has_more = True
    
    while has_more:
        url = f"https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball/scoreboard"
        params = {
            "seasontype": seasontype,
            "year": current_year,
            "limit": limit,
            "offset": offset
        }
        
        try:
            response = requests.get(url, params=params, timeout=15)
        except Exception as e:
            print(f"    Error fetching {url}: {e}")
            break
        
        if response.status_code != 200:
            print(f"    No data for seasontype={seasontype} (status {response.status_code})")
            break
        
        try:
            data = response.json()
        except Exception as e:
            print(f"    Error parsing JSON: {e}")
            break
        
        events = data.get('events', [])
        if not events:
            break
        
        print(f"    Processing {len(events)} games (offset {offset})...")
        
        for event in events:
            event_id = event.get('id')
            if event_id and event_id in seen_events:
                continue
            
            comp = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            
            if len(competitors) < 2:
                continue
            
            # Find home and away teams
            home = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            
            home_team = home.get("team", {}).get("displayName", "")
            away_team = away.get("team", {}).get("displayName", "")
            home_team_id = home.get("team", {}).get("id", "")
            away_team_id = away.get("team", {}).get("id", "")
            
            try:
                home_score = int(home.get("score", 0))
            except Exception:
                home_score = 0
            
            try:
                away_score = int(away.get("score", 0))
            except Exception:
                away_score = 0
            
            date = event.get("date", "")
            venue = comp.get("venue", {}).get("fullName", "")
            status = event.get("status", {}).get("type", {}).get("name", "")
            
            # Extract additional useful info
            game_name = event.get("name", "")
            short_name = event.get("shortName", "")
            neutral_site = comp.get("neutralSite", False)
            conference_competition = comp.get("conferenceCompetition", False)
            
            # Get conference info if available
            home_conference = home.get("team", {}).get("conferenceId", "")
            away_conference = away.get("team", {}).get("conferenceId", "")
            
            # Get rankings if available
            home_rank = home.get("curatedRank", {}).get("current", None)
            away_rank = away.get("curatedRank", {}).get("current", None)
            
            all_scores.append({
                "year": current_year,
                "season_type": seasontype_name,
                "date": date,
                "home_team": home_team,
                "away_team": away_team,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "home_score": home_score,
                "away_score": away_score,
                "home_rank": home_rank,
                "away_rank": away_rank,
                "venue": venue,
                "neutral_site": neutral_site,
                "conference_game": conference_competition,
                "home_conference": home_conference,
                "away_conference": away_conference,
                "status": status,
                "event_id": event_id,
                "game_name": game_name
            })
            
            if event_id:
                seen_events.add(event_id)
        
        # Check if there are more results
        if len(events) < limit:
            has_more = False
        else:
            offset += limit

print(f"\nTotal games collected: {len(all_scores)}")

# Create DataFrame and save
scores_df = pd.DataFrame(all_scores)

# Sort by date
scores_df['date'] = pd.to_datetime(scores_df['date'])
scores_df = scores_df.sort_values('date')

output_file = "data_files/espn_cbb_current_season.csv"
scores_df.to_csv(output_file, index=False)
print(f"Saved all scores to {output_file}")

# Print summary statistics
print(f"\nSummary:")
print(f"  Total games: {len(scores_df)}")
print(f"  Years: {scores_df['year'].min()} - {scores_df['year'].max()}")
print(f"  Season types: {scores_df['season_type'].value_counts().to_dict()}")
print(f"  Unique teams: {len(set(scores_df['home_team'].unique()) | set(scores_df['away_team'].unique()))}")
