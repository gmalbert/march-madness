"""Regenerate bracket_2026.json from bracket_2026_raw.json using actual 2026 teams."""
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from bracket_simulation import create_bracket_from_data, create_predictor_from_models
from data_tools.efficiency_loader import EfficiencyDataLoader

RAW = Path("data_files/precomputed_brackets/bracket_2026_raw.json")
OUT = Path("data_files/precomputed_brackets/bracket_2026.json")

raw = json.loads(RAW.read_text())

bracket_data = {"year": 2026, "teams": raw["teams"], "games": []}
print(f"Loaded {len(bracket_data['teams'])} teams from raw file")
for region in ["East", "West", "Midwest", "South"]:
    region_teams = [t for t in bracket_data["teams"] if t["region"] == region]
    entries = [f"{t['seed']} {t['name']}" for t in sorted(region_teams, key=lambda x: x["seed"])]
    print(f"  {region}: {entries}")

loader = EfficiencyDataLoader()
kenpom_df = loader.load_kenpom()

bracket_state, simulator = create_bracket_from_data(bracket_data)
game_predictor = create_predictor_from_models(efficiency_data=kenpom_df)
simulator.game_predictor = game_predictor

NUM_SIMS = 5000
print(f"\nRunning {NUM_SIMS} simulations...")
sim_results = simulator.simulate_bracket(bracket_state, num_simulations=NUM_SIMS)


def serialize(r):
    out = {}
    for tid, stats in r.items():
        t = stats["team"]
        out[tid] = {
            "team": {"name": t.name, "seed": t.seed, "region": t.region},
            "round_32_prob": stats.get("round_32_prob", 0.0),
            "sweet_16_prob": stats.get("sweet_16_prob", 0.0),
            "elite_8_prob": stats.get("elite_8_prob", 0.0),
            "final_four_prob": stats.get("final_four_prob", 0.0),
            "championship_prob": stats.get("championship_prob", 0.0),
            "winner_prob": stats.get("winner_prob", 0.0),
        }
    return out


output = {
    "year": 2026,
    "num_simulations": NUM_SIMS,
    "computed_at": datetime.now().isoformat(),
    "bracket_data": bracket_data,
    "simulation_results": serialize(sim_results),
}

OUT.write_text(json.dumps(output, indent=2))
print(f"\nSaved {len(sim_results)} teams to {OUT}")

# Quick sanity check
top5 = sorted(sim_results.values(), key=lambda x: x.get("winner_prob", 0), reverse=True)[:5]
print("\nTop 5 championship odds:")
for s in top5:
    t = s["team"]
    print(f"  ({t.seed}) {t.name} [{t.region}] – {s['winner_prob']*100:.1f}%")
