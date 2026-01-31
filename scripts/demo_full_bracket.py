#!/usr/bin/env python3
"""
Full tournament bracket simulation example with all rounds.
"""

from bracket_simulation import simulate_bracket, run_single_simulation
import random

random.seed(123)

print("=" * 80)
print("FULL 64-TEAM TOURNAMENT BRACKET SIMULATION")
print("=" * 80)

# Create a complete Final Four scenario
predictions = {
    "teams": [
        # East Region
        {"name": "Duke", "seed": 1, "region": "East"},
        {"name": "Baylor", "seed": 3, "region": "East"},
        {"name": "Alabama", "seed": 2, "region": "East"},
        {"name": "Clemson", "seed": 7, "region": "East"},
        
        # West Region  
        {"name": "Kansas", "seed": 1, "region": "West"},
        {"name": "Marquette", "seed": 3, "region": "West"},
        {"name": "Houston", "seed": 2, "region": "West"},
        {"name": "Indiana", "seed": 7, "region": "West"},
        
        # South Region
        {"name": "UConn", "seed": 1, "region": "South"},
        {"name": "Iowa", "seed": 3, "region": "South"},
        {"name": "Texas A&M", "seed": 2, "region": "South"},
        {"name": "Michigan", "seed": 7, "region": "South"},
        
        # Midwest Region
        {"name": "Purdue", "seed": 1, "region": "Midwest"},
        {"name": "Arkansas", "seed": 3, "region": "Midwest"},
        {"name": "Tennessee", "seed": 2, "region": "Midwest"},
        {"name": "Washington State", "seed": 7, "region": "Midwest"},
    ],
    
    # Sweet 16 (assume top 2 seeds per region advanced)
    "sweet16": [
        # East
        {"team1": "Duke", "team2": "Baylor", "team1_prob": 0.68},
        {"team1": "Alabama", "team2": "Clemson", "team1_prob": 0.72},
        # West
        {"team1": "Kansas", "team2": "Marquette", "team1_prob": 0.64},
        {"team1": "Houston", "team2": "Indiana", "team1_prob": 0.70},
        # South
        {"team1": "UConn", "team2": "Iowa", "team1_prob": 0.71},
        {"team1": "Texas A&M", "team2": "Michigan", "team1_prob": 0.66},
        # Midwest
        {"team1": "Purdue", "team2": "Arkansas", "team1_prob": 0.67},
        {"team1": "Tennessee", "team2": "Washington State", "team1_prob": 0.69},
    ],
    
    # Elite 8 (regional finals)
    "elite8": [
        {"team1": "Duke", "team2": "Alabama", "team1_prob": 0.58},
        {"team1": "Kansas", "team2": "Houston", "team1_prob": 0.52},
        {"team1": "UConn", "team2": "Texas A&M", "team1_prob": 0.63},
        {"team1": "Purdue", "team2": "Tennessee", "team1_prob": 0.56},
    ],
    
    # Final Four
    "final4": [
        {"team1": "Duke", "team2": "Kansas", "team1_prob": 0.51},
        {"team1": "UConn", "team2": "Purdue", "team1_prob": 0.55},
    ],
    
    # Championship
    "championship": [
        {"team1": "Duke", "team2": "UConn", "team1_prob": 0.48},
    ],
}

print("\n🏀 Running 25,000 simulations...\n")

# Run larger simulation for more accurate probabilities
results = simulate_bracket(predictions, num_sims=25000)

# Sort by winner probability
teams_sorted = sorted(
    results.items(),
    key=lambda x: x[1]['winner_pct'],
    reverse=True
)

print(f"{'Seed':<6} {'Team':<18} {'Sweet 16':<10} {'Elite 8':<10} {'Final 4':<10} {'Title Game':<12} {'Champion':<10}")
print("-" * 88)

for team_name, probs in teams_sorted:
    # Find seed
    team_data = next((t for t in predictions["teams"] if t["name"] == team_name), {})
    seed = team_data.get("seed", "?")
    region = team_data.get("region", "?")[:1]  # First letter
    
    # For this simplified bracket, sweet 16 participants are in the matchups
    sweet16_teams = set()
    for matchup in predictions.get("sweet16", []):
        sweet16_teams.add(matchup["team1"])
        sweet16_teams.add(matchup["team2"])
    
    sweet16_prob = 1.0 if team_name in sweet16_teams else 0.0
    
    # Elite 8 participants
    elite8_teams = set()
    for matchup in predictions.get("elite8", []):
        elite8_teams.add(matchup["team1"])
        elite8_teams.add(matchup["team2"])
    
    elite8_prob = 1.0 if team_name in elite8_teams else 0.0
    
    print(f"{f'#{seed} {region}':<6} {team_name:<18} "
          f"{sweet16_prob:>8.1%}  "
          f"{elite8_prob:>8.1%}  "
          f"{probs['final_four_pct']:>8.1%}  "
          f"{probs['championship_pct']:>10.1%}  "
          f"{probs['winner_pct']:>8.1%}")

print("\n" + "=" * 88)
print("BRACKET ANALYSIS")
print("=" * 88)

# Top 4 championship contenders
print("\n🏆 Top Championship Contenders:\n")
for i, (team, probs) in enumerate(teams_sorted[:4], 1):
    print(f"  {i}. {team}: {probs['winner_pct']:.1%} to win it all")

# Find the chalk (all #1 seeds making Final Four)
one_seeds = [t["name"] for t in predictions["teams"] if t.get("seed") == 1]
one_seed_ff_probs = [results[team]['final_four_pct'] for team in one_seeds]
all_ones_ff = 1.0
for prob in one_seed_ff_probs:
    all_ones_ff *= prob

print(f"\n📊 Probability all #1 seeds reach Final Four: {all_ones_ff:.2%}")

# Most likely Final Four
print("\n🎯 Most Likely Final Four Appearance:\n")
ff_sorted = sorted(results.items(), key=lambda x: x[1]['final_four_pct'], reverse=True)
for team, probs in ff_sorted[:4]:
    team_data = next((t for t in predictions["teams"] if t["name"] == team), {})
    seed = team_data.get("seed", "?")
    region = team_data.get("region", "?")
    print(f"  {team} (#{seed} {region}): {probs['final_four_pct']:.1%}")

# Cinderella potential (lower seeds with decent chances)
print("\n🌟 Cinderella Stories (Lower Seeds with >5% Title Chance):\n")
cinderellas = [
    (team, probs, next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed", 1))
    for team, probs in results.items()
    if probs['winner_pct'] > 0.05 and next((t for t in predictions["teams"] if t["name"] == team), {}).get("seed", 1) > 1
]
cinderellas.sort(key=lambda x: x[1]['winner_pct'], reverse=True)

if cinderellas:
    for team, probs, seed in cinderellas:
        print(f"  #{seed} {team}: {probs['winner_pct']:.1%} to win")
else:
    print("  No major Cinderella candidates in this bracket")

# Championship matchup probabilities
print("\n🥇 Most Likely Championship Matchups:\n")

# We need to track this during simulation - for now, show implied matchups
print("  Based on individual probabilities:")
top_4 = teams_sorted[:4]
for i in range(len(top_4)):
    for j in range(i+1, len(top_4)):
        team1_name, team1_probs = top_4[i]
        team2_name, team2_probs = top_4[j]
        # Rough estimate: P(both reach finals)
        implied_prob = team1_probs['championship_pct'] * team2_probs['championship_pct']
        if implied_prob > 0.01:  # >1% chance
            print(f"    {team1_name} vs {team2_name}: ~{implied_prob:.1%}")

print("\n" + "=" * 88)
