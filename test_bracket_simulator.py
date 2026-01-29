#!/usr/bin/env python3
"""
Test the simplified Bracket Simulator from the roadmap.

Demonstrates Monte Carlo simulation of tournament outcomes.
"""

from bracket_simulation import simulate_bracket, run_single_simulation
import random

# Set seed for reproducible results in demo
random.seed(42)

print("=" * 80)
print("BRACKET SIMULATOR - ROADMAP IMPLEMENTATION")
print("=" * 80)

# Create sample tournament predictions
# Simplified 8-team bracket for demonstration
predictions = {
    "teams": [
        {"name": "Duke", "seed": 1, "region": "East", "stats": {"net_efficiency": 25.0}},
        {"name": "Alabama", "seed": 2, "region": "East", "stats": {"net_efficiency": 22.0}},
        {"name": "Kansas", "seed": 1, "region": "West", "stats": {"net_efficiency": 24.0}},
        {"name": "Houston", "seed": 2, "region": "West", "stats": {"net_efficiency": 23.0}},
        {"name": "UConn", "seed": 1, "region": "South", "stats": {"net_efficiency": 26.0}},
        {"name": "Tennessee", "seed": 2, "region": "South", "stats": {"net_efficiency": 21.0}},
        {"name": "Purdue", "seed": 1, "region": "Midwest", "stats": {"net_efficiency": 23.5}},
        {"name": "Arizona", "seed": 2, "region": "Midwest", "stats": {"net_efficiency": 20.0}},
    ],
    
    # Elite 8 matchups (simplified - assume favorites advanced)
    "elite8": [
        {"team1": "Duke", "team2": "Alabama", "team1_prob": 0.65},
        {"team1": "Kansas", "team2": "Houston", "team1_prob": 0.55},
        {"team1": "UConn", "team2": "Tennessee", "team1_prob": 0.70},
        {"team1": "Purdue", "team2": "Arizona", "team1_prob": 0.62},
    ],
    
    # Final Four (winners from Elite 8)
    "final4": [
        {"team1": "Duke", "team2": "Kansas", "team1_prob": 0.52},
        {"team1": "UConn", "team2": "Purdue", "team1_prob": 0.58},
    ],
    
    # Championship
    "championship": [
        {"team1": "Duke", "team2": "UConn", "team1_prob": 0.48},
    ],
}

print("\n--- SINGLE SIMULATION EXAMPLE ---\n")

# Run a single simulation
bracket = run_single_simulation(predictions)
print("Single Tournament Outcome:")
print(f"  Elite 8 Winners: {bracket.get('elite8_winners', [])}")
print(f"  Final Four: {bracket.get('final_four', [])}")
print(f"  Championship Game: {bracket.get('championship', [])}")
print(f"  Champion: {bracket.get('winner', 'N/A')}")

print("\n" + "-" * 80)
print("\n--- MONTE CARLO SIMULATION (10,000 iterations) ---\n")

# Run full Monte Carlo simulation
results = simulate_bracket(predictions, num_sims=10000)

# Sort teams by championship probability
teams_sorted = sorted(
    results.items(), 
    key=lambda x: x[1]['winner_pct'], 
    reverse=True
)

print("Tournament Probabilities:\n")
print(f"{'Team':<15} {'Final Four':<12} {'Championship':<14} {'Winner':<10}")
print("-" * 52)

for team, probs in teams_sorted:
    print(f"{team:<15} "
          f"{probs['final_four_pct']:>10.1%}  "
          f"{probs['championship_pct']:>12.1%}  "
          f"{probs['winner_pct']:>8.1%}")

print("\n" + "=" * 80)
print("KEY INSIGHTS")
print("=" * 80)

# Find most likely champion
most_likely_champion = teams_sorted[0]
print(f"\nMost Likely Champion: {most_likely_champion[0]}")
print(f"  Win Probability: {most_likely_champion[1]['winner_pct']:.1%}")
print(f"  Championship Game Probability: {most_likely_champion[1]['championship_pct']:.1%}")

# Find biggest Final Four lock
final_four_sorted = sorted(
    results.items(), 
    key=lambda x: x[1]['final_four_pct'], 
    reverse=True
)
biggest_lock = final_four_sorted[0]
print(f"\nBiggest Final Four Lock: {biggest_lock[0]}")
print(f"  Final Four Probability: {biggest_lock[1]['final_four_pct']:.1%}")

# Identify upset potential
print("\nUpset Scenarios:")
for team, probs in teams_sorted:
    # Check if lower seed has decent chance
    team_data = next((t for t in predictions["teams"] if t["name"] == team), None)
    if team_data and team_data.get("seed", 1) > 1:
        if probs['winner_pct'] > 0.10:  # >10% chance to win it all
            print(f"  {team} (#{team_data['seed']} seed) has {probs['winner_pct']:.1%} chance to win")

print("\n" + "=" * 80)
print("SIMULATION STATISTICS")
print("=" * 80)

# Calculate some interesting stats
total_final_four_prob = sum(p['final_four_pct'] for p in results.values())
total_champion_prob = sum(p['winner_pct'] for p in results.values())

print(f"\nTotal Final Four probability (should be ~4.0): {total_final_four_prob:.2f}")
print(f"Total Champion probability (should be ~1.0): {total_champion_prob:.3f}")

# Identify favorite vs field scenarios
top_team = teams_sorted[0]
field_prob = 1.0 - top_team[1]['winner_pct']
print(f"\nFavorite vs Field:")
print(f"  {top_team[0]}: {top_team[1]['winner_pct']:.1%}")
print(f"  The Field: {field_prob:.1%}")

print("\n" + "=" * 80)
print("USAGE EXAMPLE")
print("=" * 80)
print("""
# Basic usage:
from bracket_simulation import simulate_bracket

predictions = {
    "teams": [...],  # List of team dictionaries
    "elite8": [...], # Matchups with team1_prob
    "final4": [...],
    "championship": [...]
}

results = simulate_bracket(predictions, num_sims=10000)

for team, probs in results.items():
    print(f"{team}: {probs['winner_pct']:.1%} to win")
""")
