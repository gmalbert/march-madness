#!/usr/bin/env python3
"""
Fetch the 2026 NCAA Tournament bracket from ESPN's public API.

Saves to data_files/precomputed_brackets/bracket_2026_raw.json (team list)
and enriches with KenPom/BartTorvik efficiency data, then calls the
bracket simulation to produce bracket_2026.json.

Usage:
    python scripts/fetch_2026_bracket.py
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime, timedelta

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

DATA_DIR = Path("data_files")
BRACKETS_DIR = DATA_DIR / "precomputed_brackets"
BRACKETS_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_seed_from_summary(event_id: str) -> dict:
    """
    Use the ESPN event summary endpoint to get seeds for each team in a game.
    Returns {team_id: seed} or empty dict on failure.
    """
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/"
        "mens-college-basketball/summary"
    )
    try:
        r = requests.get(url, params={"event": event_id}, timeout=15)
        if r.status_code != 200:
            return {}
        data = r.json()
        header = data.get("header", {})
        comp = header.get("competitions", [{}])[0]
        result = {}
        for c in comp.get("competitors", []):
            tid = c.get("id", "") or c.get("team", {}).get("id", "")
            rank = c.get("rank")  # In NCAA tournament context this is the seed
            if tid and rank is not None:
                try:
                    result[str(tid)] = int(rank)
                except (TypeError, ValueError):
                    pass
        return result
    except Exception:
        return {}


def fetch_tournament_teams() -> dict:
    """Fetch all 2026 tournament teams from ESPN, using March 17-23 game slate.

    For each game we call the event summary endpoint to get the actual
    tournament seed (stored in the 'rank' field of header competitors).
    """
    all_teams = {}
    seen_events = set()

    # First Four: March 17-18; Round of 64: March 19-20; Round of 32: March 21-22
    for offset in range(7):
        dt = datetime(2026, 3, 17) + timedelta(days=offset)
        date_str = dt.strftime("%Y%m%d")
        url = (
            "https://site.api.espn.com/apis/site/v2/sports/basketball/"
            "mens-college-basketball/scoreboard"
        )
        try:
            r = requests.get(url, params={"dates": date_str, "limit": 100}, timeout=15)
        except Exception as exc:
            print(f"  Network error for {date_str}: {exc}")
            continue

        if r.status_code != 200:
            print(f"  No data for {date_str} (HTTP {r.status_code})")
            continue

        events = r.json().get("events", [])
        for e in events:
            if e["id"] in seen_events:
                continue
            comp = e.get("competitions", [{}])[0]
            notes = comp.get("notes", [{}])
            headline = notes[0].get("headline", "") if notes else ""
            if "NCAA" not in headline:
                continue
            seen_events.add(e["id"])

            # Determine region from headline
            region = "TBD"
            for reg in ["East", "West", "Midwest", "South"]:
                if reg in headline:
                    region = reg
                    break

            # Round label
            round_label = "1st Round"
            if "First Four" in headline:
                round_label = "First Four"
            elif "2nd Round" in headline:
                round_label = "2nd Round"

            # Fetch seeds from the event summary (more reliable)
            seed_map = _fetch_seed_from_summary(e["id"])

            competitors = comp.get("competitors", [])
            for td in competitors:
                team = td.get("team", {})
                team_id = team.get("id", "")
                team_name = team.get("displayName", "")

                # Prefer seed from summary; fall back to scoreboard fields
                seed = seed_map.get(str(team_id), 0)
                if not seed:
                    seed = td.get("seed") or 0
                    try:
                        seed = int(seed)
                    except (TypeError, ValueError):
                        seed = 0

                if team_id and team_id not in all_teams:
                    all_teams[team_id] = {
                        "id": team_id,
                        "name": team_name,
                        "seed": seed,
                        "region": region,
                        "round_label": round_label,
                        "stats": {},
                    }
                elif team_id:
                    # Update seed / region if we now have better data
                    if all_teams[team_id]["seed"] == 0 and seed:
                        all_teams[team_id]["seed"] = seed
                    if all_teams[team_id]["region"] == "TBD" and region != "TBD":
                        all_teams[team_id]["region"] = region

    return all_teams


def enrich_with_efficiency(teams: dict) -> dict:
    """Add KenPom / BartTorvik stats to each team entry."""
    from data_tools.efficiency_loader import EfficiencyDataLoader
    from predictions import normalize_team_name

    loader = EfficiencyDataLoader()

    kenpom_df = None
    bart_df = None
    try:
        kenpom_df = loader.load_kenpom()
        print(f"  KenPom: {len(kenpom_df)} teams loaded")
    except Exception as exc:
        print(f"  Warning: KenPom load failed: {exc}")

    try:
        bart_df = loader.load_barttorvik()
        print(f"  BartTorvik: {len(bart_df)} teams loaded")
    except Exception as exc:
        print(f"  Warning: BartTorvik load failed: {exc}")

    for team_id, team in teams.items():
        name = team["name"]
        norm = normalize_team_name(name)
        stats = {}

        if kenpom_df is not None and len(kenpom_df) > 0:
            team_col = "TeamName" if "TeamName" in kenpom_df.columns else "Team"
            for lookup in [name, norm]:
                match = kenpom_df[kenpom_df[team_col] == lookup]
                if not match.empty:
                    row = match.iloc[0]
                    stats.update(
                        {
                            "net_efficiency": float(row.get("NetRtg", 0) or 0),
                            "off_efficiency": float(row.get("ORtg", 0) or 0),
                            "def_efficiency": float(row.get("DRtg", 0) or 0),
                            "tempo": float(row.get("AdjT", 70) or 70),
                            "luck": float(row.get("Luck", 0) or 0),
                            "sos": float(row.get("SOS_NetRtg", 0) or 0),
                        }
                    )
                    break

        if bart_df is not None and len(bart_df) > 0:
            team_col = "canonical_team" if "canonical_team" in bart_df.columns else "Team"
            for lookup in [name, norm]:
                match = bart_df[bart_df[team_col] == lookup]
                if not match.empty:
                    row = match.iloc[0]
                    # column names vary; try common ones
                    adj_oe = row.get("AdjOE") or row.get("adj_oe") or row.get("AdjO") or 0
                    adj_de = row.get("AdjDE") or row.get("adj_de") or row.get("AdjD") or 0
                    stats.update(
                        {
                            "bart_adj_oe": float(adj_oe or 0),
                            "bart_adj_de": float(adj_de or 0),
                        }
                    )
                    break

        team["stats"] = stats

    return teams


def build_bracket_data(teams: dict) -> dict:
    """Convert teams dict into bracket_data for bracket_simulation.

    The simulation engine (_pair_region_teams) requires exactly 16 teams with
    unique seeds 1-16 per region.  First Four games produce two teams sharing
    the same (region, seed) slot.  We resolve each First Four slot by keeping
    the team with the higher KenPom net_efficiency.  The full 70-team list is
    preserved in 'all_teams' for display purposes.
    """
    from collections import defaultdict

    # Group by (region, seed) to find First Four duplicates
    slot_map: dict = defaultdict(list)
    for team in teams.values():
        key = (team["region"], team["seed"])
        slot_map[key].append(team)

    first_four_results = []
    simulation_teams = []

    for (region, seed), group in slot_map.items():
        if len(group) == 1:
            simulation_teams.append(group[0])
        else:
            # First Four: two teams competing for one slot – pick the better one
            ranked = sorted(
                group,
                key=lambda t: t.get("stats", {}).get("net_efficiency", 0) or 0,
                reverse=True,
            )
            winner, loser = ranked[0], ranked[1]
            entry = dict(winner)
            entry["first_four_opponent"] = loser["name"]
            simulation_teams.append(entry)
            first_four_results.append(
                {
                    "region": region,
                    "seed": seed,
                    "kept": winner["name"],
                    "dropped": loser["name"],
                    "winner_eff": winner.get("stats", {}).get("net_efficiency", 0),
                    "loser_eff": loser.get("stats", {}).get("net_efficiency", 0),
                }
            )
            print(
                f"  First Four ({region} {seed}-seed): "
                f"{winner['name']} ({winner.get('stats',{}).get('net_efficiency',0):.1f}) "
                f"beats {loser['name']} ({loser.get('stats',{}).get('net_efficiency',0):.1f})"
            )

    # Remove seed-0 TBD placeholders and teams with undetermined region
    before = len(simulation_teams)
    simulation_teams = [
        t for t in simulation_teams
        if t.get("seed", 0) > 0 and t.get("region", "TBD") in ("East", "West", "Midwest", "South")
    ]
    if before != len(simulation_teams):
        print(f"  Removed {before - len(simulation_teams)} placeholder/TBD teams")

    # Log region sizes so we can verify each has exactly 16
    from collections import Counter
    region_counts = Counter(t["region"] for t in simulation_teams)
    print(f"  Simulation bracket team counts: {dict(region_counts)}")

    return {
        "year": 2026,
        "teams": simulation_teams,   # 64 teams (16 per region) for simulation
        "all_teams": list(teams.values()),  # full 70-team list for display
        "first_four": first_four_results,
        "games": [],
    }


def main():
    print("=" * 60)
    print("Fetching 2026 NCAA Tournament bracket from ESPN")
    print("=" * 60)

    print("\nStep 1: Fetching tournament teams from ESPN scoreboard...")
    teams = fetch_tournament_teams()
    print(f"  Found {len(teams)} unique teams")

    if len(teams) < 32:
        print(
            f"  WARNING: Expected ~68 teams, got {len(teams)}. "
            "ESPN may not have all data yet – proceeding with available teams."
        )

    # Print sorted bracket
    print("\n  Team list:")
    for t in sorted(teams.values(), key=lambda x: (x["region"], x["seed"])):
        print(f"    {t['region']:10} Seed {t['seed']:2}: {t['name']}")

    print("\nStep 2: Enriching with efficiency data (KenPom / BartTorvik)...")
    teams = enrich_with_efficiency(teams)
    enriched = sum(1 for t in teams.values() if t["stats"])
    print(f"  Enriched {enriched}/{len(teams)} teams with efficiency data")

    print("\nStep 3: Building bracket data structure...")
    bracket_data = build_bracket_data(teams)

    # Save raw bracket data
    raw_path = BRACKETS_DIR / "bracket_2026_raw.json"
    with open(raw_path, "w") as f:
        json.dump(bracket_data, f, indent=2)
    print(f"  Saved raw bracket to {raw_path}")

    print("\nStep 4: Running Monte Carlo bracket simulation (10,000 sims)...")
    from bracket_simulation import (
        create_bracket_from_data,
        create_predictor_from_models,
    )
    from data_tools.efficiency_loader import EfficiencyDataLoader

    efficiency_loader = EfficiencyDataLoader()
    kenpom_df = efficiency_loader.load_kenpom()

    bracket_state, simulator = create_bracket_from_data(bracket_data)
    game_predictor = create_predictor_from_models(efficiency_data=kenpom_df)
    simulator.game_predictor = game_predictor

    num_sims = 10_000
    simulation_results = simulator.simulate_bracket(bracket_state, num_simulations=num_sims)

    print(f"  Simulation complete: {len(simulation_results)} teams tracked")

    # Serialize results
    serialized = {}
    for team_id, stats in simulation_results.items():
        team = stats["team"]
        serialized[team_id] = {
            "team": {
                "name": team.name,
                "seed": team.seed,
                "region": team.region,
            },
            "round_32_prob": stats.get("round_32_prob", 0.0),
            "sweet_16_prob": stats.get("sweet_16_prob", 0.0),
            "elite_8_prob": stats.get("elite_8_prob", 0.0),
            "final_four_prob": stats.get("final_four_prob", 0.0),
            "championship_prob": stats.get("championship_prob", 0.0),
            "winner_prob": stats.get("winner_prob", 0.0),
        }

    output = {
        "year": 2026,
        "num_simulations": num_sims,
        "computed_at": datetime.now().isoformat(),
        "bracket_data": bracket_data,
        "simulation_results": serialized,
    }

    out_path = BRACKETS_DIR / "bracket_2026.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved simulation results to {out_path}")

    # Print top championship contenders
    print("\nTop Championship Contenders:")
    top = sorted(serialized.items(), key=lambda x: x[1]["winner_prob"], reverse=True)[:10]
    for tid, stats in top:
        t = stats["team"]
        print(
            f"  ({t['seed']}) {t['name']:30} {t['region']:10} "
            f"Win%={stats['winner_prob']:.1%}  FF%={stats['final_four_prob']:.1%}"
        )

    print("\nDone! 2026 bracket simulation complete.")


if __name__ == "__main__":
    main()
