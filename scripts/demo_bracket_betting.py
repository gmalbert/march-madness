#!/usr/bin/env python3
"""
Integration example: Using Bracket Simulator with Tournament Predictor.

Shows how to combine the upset prediction model with bracket simulation
to generate realistic tournament outcome probabilities.
"""

from bracket_simulation import simulate_bracket, run_single_simulation
from upset_prediction import UpsetPredictor
import random

random.seed(2025)

print("=" * 80)
print("BRACKET SIMULATOR + UPSET PREDICTOR INTEGRATION")
print("=" * 80)

# Create realistic tournament scenario
# This demonstrates using upset probabilities in bracket simulation

# Sample Final Four scenario with upset-adjusted probabilities
predictions = {
    "teams": [
        {"name": "Duke", "seed": 1, "region": "East", "stats": {"net_efficiency": 25.0}},
        {"name": "Iowa State", "seed": 5, "region": "East", "stats": {"net_efficiency": 18.0}},  # Upset candidate
        {"name": "Kansas", "seed": 1, "region": "West", "stats": {"net_efficiency": 24.0}},
        {"name": "Creighton", "seed": 6, "region": "West", "stats": {"net_efficiency": 17.5}},  # Upset candidate
        {"name": "UConn", "seed": 1, "region": "South", "stats": {"net_efficiency": 26.0}},
        {"name": "San Diego State", "seed": 5, "region": "South", "stats": {"net_efficiency": 18.5}},  # Upset candidate
        {"name": "Purdue", "seed": 1, "region": "Midwest", "stats": {"net_efficiency": 23.5}},
        {"name": "Gonzaga", "seed": 3, "region": "Midwest", "stats": {"net_efficiency": 21.0}},
    ],
    
    # Elite 8 matchups with upset potential
    # Note: Lower probabilities for favorites due to upset risk
    "elite8": [
        {"team1": "Duke", "team2": "Iowa State", "team1_prob": 0.62},  # 5-seed has upset chance
        {"team1": "Kansas", "team2": "Creighton", "team1_prob": 0.65}, # 6-seed dangerous
        {"team1": "UConn", "team2": "San Diego State", "team1_prob": 0.68},  # 5-seed with potential
        {"team1": "Purdue", "team2": "Gonzaga", "team1_prob": 0.58},  # 3-seed close matchup
    ],
    
    # Final Four
    "final4": [
        {"team1": "Duke", "team2": "Kansas", "team1_prob": 0.51},
        {"team1": "UConn", "team2": "Purdue", "team1_prob": 0.56},
    ],
    
    # Championship
    "championship": [
        {"team1": "Duke", "team2": "UConn", "team1_prob": 0.47},
    ],
}

print("\n📊 SCENARIO: Final Four Weekend with Upset Candidates")
print("=" * 80)
print("\nThis simulation includes several potential upset scenarios:")
print("  • #5 Iowa State vs #1 Duke (Elite 8)")
print("  • #6 Creighton vs #1 Kansas (Elite 8)")
print("  • #5 San Diego State vs #1 UConn (Elite 8)")
print("  • #3 Gonzaga vs #1 Purdue (Elite 8)")
print()

# Run simulation
print("🔄 Running 50,000 Monte Carlo simulations...\n")
results = simulate_bracket(predictions, num_sims=50000)

# Display results
teams_sorted = sorted(
    results.items(),
    key=lambda x: x[1]['winner_pct'],
    reverse=True
)

print(f"{'Seed':<8} {'Team':<20} {'Final 4':<12} {'Title Game':<14} {'Champion':<12} {'Upset Factor'}")
print("-" * 90)

for team_name, probs in teams_sorted:
    team_data = next((t for t in predictions["teams"] if t["name"] == team_name), {})
    seed = team_data.get("seed", "?")
    region = team_data.get("region", "?")[:1]
    
    # Calculate "upset factor" - how much they're overperforming seed
    expected_title_prob = {1: 0.25, 2: 0.15, 3: 0.10, 4: 0.05, 5: 0.03, 6: 0.02}
    expected = expected_title_prob.get(seed, 0.01)
    upset_factor = (probs['winner_pct'] / expected) if expected > 0 else 1.0
    
    upset_indicator = "🔥" if upset_factor > 1.5 else ("⚡" if upset_factor > 1.2 else "")
    
    print(f"#{seed} {region:<5} {team_name:<20} "
          f"{probs['final_four_pct']:>10.1%}  "
          f"{probs['championship_pct']:>12.1%}  "
          f"{probs['winner_pct']:>10.1%}  "
          f"{upset_factor:>5.2f}x {upset_indicator}")

print("\n" + "=" * 90)
print("KEY INSIGHTS")
print("=" * 90)

# Find the favorites
print("\n🏆 Championship Favorites:\n")
for i, (team, probs) in enumerate(teams_sorted[:3], 1):
    team_data = next((t for t in predictions["teams"] if t["name"] == team), {})
    seed = team_data.get("seed")
    print(f"  {i}. {team} (#{seed} seed): {probs['winner_pct']:.1%}")

# Upset candidates
print("\n💥 Best Upset Candidates (Lower Seeds with Solid Chances):\n")
upsets = [
    (team, probs, next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed", 1))
    for team, probs in results.items()
    if next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed", 1) >= 3
    and probs['winner_pct'] > 0.02
]
upsets.sort(key=lambda x: x[1]['winner_pct'], reverse=True)

for team, probs, seed in upsets[:3]:
    ff_prob = probs['final_four_pct']
    title_prob = probs['winner_pct']
    print(f"  #{seed} {team}:")
    print(f"    Final Four: {ff_prob:.1%}")
    print(f"    Win Title: {title_prob:.1%}")

# Chalk probability (all 1-seeds in Final Four)
print("\n📈 Bracket Chaos Metrics:\n")
one_seeds = [team for team, _ in results.items() 
             if next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed") == 1]

all_ones_ff = 1.0
for team in one_seeds:
    all_ones_ff *= results[team]['final_four_pct']

print(f"  Probability of all #1 seeds in Final Four: {all_ones_ff:.2%}")
print(f"  Probability of at least one upset in Final Four: {1 - all_ones_ff:.1%}")

# Most likely upset winner
lower_seeds = [
    (team, probs, next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed", 1))
    for team, probs in results.items()
    if next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed", 1) > 2
]
if lower_seeds:
    most_likely_upset = max(lower_seeds, key=lambda x: x[1]['winner_pct'])
    print(f"\n🌟 Most Likely Cinderella Champion: #{most_likely_upset[2]} {most_likely_upset[0]}")
    print(f"   Odds: {most_likely_upset[1]['winner_pct']:.1%}")

# Expected value analysis
print("\n💰 Betting Value Analysis:\n")
print("  If betting on outright champion at equal odds:")
for team, probs in teams_sorted[:5]:
    implied_odds = 1 / probs['winner_pct'] if probs['winner_pct'] > 0 else 999
    american_odds = (implied_odds - 1) * 100 if implied_odds >= 2 else -100 / (implied_odds - 1)
    print(f"  {team:<20} Win prob: {probs['winner_pct']:>6.1%}  Fair odds: {american_odds:>+7.0f}")

print("\n" + "=" * 90)
print("USAGE IN BETTING STRATEGY")
print("=" * 90)
print("""
This simulation helps identify:

1. VALUE BETS: Teams with higher win probability than their odds suggest
   - Look for teams where simulation % > implied probability from odds

2. UPSET SPECIAL: Lower seeds with realistic championship chances
   - These often have long odds but decent probability in simulation
   - Example: #5 or #6 seed with 2-5% chance at good odds

3. FINAL FOUR FUTURES: Teams with high Final Four probability
   - Safer bet than outright winner
   - Look for >50% Final Four probability

4. HEDGE OPPORTUNITIES: Identify when to hedge championship futures
   - If you have a future on a team, simulation shows when to hedge

5. BRACKET STRATEGY: Understand chalk vs. upset balance
   - Use probabilities to optimize bracket picks by round
""")

print("=" * 90)
