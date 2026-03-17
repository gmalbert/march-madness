"""
2026 NCAA Men's March Madness Bracket Scraper
Uses ESPN's unofficial public API + web scraping fallback.
No API key required.
"""

import requests
import json
from pprint import pprint

# ─────────────────────────────────────────────
# OPTION 1: ESPN Unofficial API (best approach)
# Returns live bracket data including seeds,
# regions, matchups, and scores.
# ─────────────────────────────────────────────

BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/mens-college-basketball"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; bracket-scraper/1.0)"
}


def fetch_tournament_bracket():
    """
    Fetch the NCAA tournament bracket via ESPN's scoreboard API.
    group=100 ensures all tournament games are returned.
    """
    # Tournament games span multiple days — fetch by date range
    # First Four: March 17-18 | First Round: March 19-20
    tournament_dates = [
        "20260317", "20260318",  # First Four
        "20260319", "20260320",  # First Round
        "20260321", "20260322",  # Second Round
    ]

    all_games = []

    for date in tournament_dates:
        url = f"{BASE}/scoreboard"
        params = {
            "dates": date,
            "groups": 100,   # include all games
            "limit": 100,
        }
        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        events = data.get("events", [])
        for event in events:
            game = parse_game(event)
            if game:
                all_games.append(game)

        print(f"  {date}: found {len(events)} games")

    return all_games


def parse_game(event):
    """Parse a single game/event from ESPN API response."""
    competitions = event.get("competitions", [])
    if not competitions:
        return None

    comp = competitions[0]
    competitors = comp.get("competitors", [])

    teams = []
    for c in competitors:
        team_info = c.get("team", {})
        curRecord = c.get("records", [{}])[0].get("summary", "N/A") if c.get("records") else "N/A"
        teams.append({
            "team": team_info.get("displayName", "Unknown"),
            "abbreviation": team_info.get("abbreviation", ""),
            "seed": c.get("curatedRank", {}).get("current") or c.get("seed", "N/A"),
            "score": c.get("score", "TBD"),
            "winner": c.get("winner", False),
            "record": curRecord,
            "home_away": c.get("homeAway", ""),
        })

    # Tournament round info
    status = event.get("status", {})
    situation = comp.get("situation", {})
    notes = comp.get("notes", [])
    round_name = ""
    region = ""
    for note in notes:
        text = note.get("text", "")
        if "Region" in text or "First Four" in text or "Final Four" in text:
            region = text
        if not round_name:
            round_name = text

    return {
        "game_id": event.get("id"),
        "name": event.get("name", ""),
        "date": event.get("date", ""),
        "round": round_name,
        "region": region,
        "venue": comp.get("venue", {}).get("fullName", "TBD"),
        "city": comp.get("venue", {}).get("address", {}).get("city", ""),
        "state": comp.get("venue", {}).get("address", {}).get("state", ""),
        "status": status.get("type", {}).get("description", ""),
        "teams": teams,
    }


def fetch_rankings():
    """
    Fetch current AP / Coaches Poll rankings via ESPN API.
    """
    url = f"{BASE}/rankings"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    rankings_output = {}

    for poll in data.get("rankings", []):
        poll_name = poll.get("name", "Unknown Poll")
        entries = []
        for rank in poll.get("ranks", []):
            team = rank.get("team", {})
            entries.append({
                "rank": rank.get("current"),
                "previous_rank": rank.get("previous"),
                "team": team.get("displayName", "Unknown"),
                "conference": team.get("conferenceId", ""),
                "record": rank.get("recordSummary", ""),
                "points": rank.get("points", 0),
                "first_place_votes": rank.get("firstPlaceVotes", 0),
            })
        rankings_output[poll_name] = entries

    return rankings_output


def fetch_teams_in_tournament():
    """
    Use ESPN's tournament-specific group endpoint to get seeded teams.
    Group 100 = NCAA Tournament field.
    """
    url = f"{BASE}/teams"
    params = {"groups": 100, "limit": 100}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json().get("sports", [{}])[0].get("leagues", [{}])[0].get("teams", [])


def print_bracket(games):
    """Pretty-print the bracket grouped by round/region."""
    from collections import defaultdict
    by_round = defaultdict(list)
    for g in games:
        by_round[g["round"] or "Unknown Round"].append(g)

    for round_name, round_games in sorted(by_round.items()):
        print(f"\n{'='*60}")
        print(f"  {round_name.upper()}")
        print(f"{'='*60}")
        for g in round_games:
            t = g["teams"]
            if len(t) == 2:
                t1, t2 = t[0], t[1]
                seed1 = f"({t1['seed']})" if t1['seed'] != 'N/A' else ""
                seed2 = f"({t2['seed']})" if t2['seed'] != 'N/A' else ""
                score1 = t1['score'] if g['status'] not in ('Scheduled', '') else ""
                score2 = t2['score'] if g['status'] not in ('Scheduled', '') else ""
                w1 = " ✓" if t1['winner'] else ""
                w2 = " ✓" if t2['winner'] else ""
                print(f"  {seed1:5} {t1['team']:30} {score1}{w1}")
                print(f"       vs.")
                print(f"  {seed2:5} {t2['team']:30} {score2}{w2}")
                print(f"  📍 {g['venue']} — {g['city']}, {g['state']}")
                print(f"  🗓  {g['date'][:10]}  |  {g['status']}")
                print()


def print_rankings(rankings):
    """Pretty-print AP/Coaches Poll rankings."""
    for poll_name, ranks in rankings.items():
        print(f"\n{'='*60}")
        print(f"  {poll_name.upper()}")
        print(f"{'='*60}")
        print(f"  {'Rank':<6} {'Team':<35} {'Record':<12} {'Points'}")
        print(f"  {'-'*4:<6} {'-'*33:<35} {'-'*10:<12} {'-'*6}")
        for r in ranks[:25]:
            fv = f" ({r['first_place_votes']} FPV)" if r['first_place_votes'] else ""
            print(f"  {r['rank']:<6} {r['team']:<35} {r['record']:<12} {r['points']}{fv}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n🏀 2026 NCAA Men's March Madness Bracket + Rankings")
    print("=" * 60)

    # 1. Rankings
    print("\n📊 Fetching current rankings...")
    try:
        rankings = fetch_rankings()
        print_rankings(rankings)

        # Save rankings to JSON
        with open("rankings_2026.json", "w") as f:
            json.dump(rankings, f, indent=2)
        print("\n✅ Rankings saved to rankings_2026.json")
    except Exception as e:
        print(f"  ⚠️  Could not fetch rankings: {e}")

    # 2. Bracket / Games
    print("\n\n🏆 Fetching tournament bracket...")
    try:
        games = fetch_tournament_bracket()
        print(f"\nTotal tournament games found: {len(games)}\n")
        print_bracket(games)

        # Save bracket to JSON
        with open("bracket_2026.json", "w") as f:
            json.dump(games, f, indent=2)
        print("\n✅ Bracket saved to bracket_2026.json")
    except Exception as e:
        print(f"  ⚠️  Could not fetch bracket: {e}")

    # 3. Seeded teams list
    print("\n\n📋 Fetching seeded tournament teams...")
    try:
        teams = fetch_teams_in_tournament()
        print(f"  Found {len(teams)} teams in the tournament field.")
        for t in teams:
            team = t.get("team", {})
            print(f"  • {team.get('displayName', '?'):35} {team.get('abbreviation', '')}")

        with open("teams_2026.json", "w") as f:
            json.dump(teams, f, indent=2)
        print("\n✅ Teams list saved to teams_2026.json")
    except Exception as e:
        print(f"  ⚠️  Could not fetch teams: {e}")