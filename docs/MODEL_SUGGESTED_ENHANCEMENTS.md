# Bracket Oracle — Model Suggested Enhancements

## Priority 1: Efficiency Rating Model

### KenPom + BartTorvik Blend
- Current models rely on one source at a time. Blend KenPom AdjEM with BartTorvik Barthag for a more robust efficiency rating.
- Use `blend = 0.5 * kenpom_adjEM + 0.5 * barttorvik_barthag` as a single composite efficiency input.

### Tempo Interaction Feature
- Fast-paced teams playing a slow-paced defence create unique scoring environments.
- Add `|home_tempo - away_tempo|` as a variance-increasing feature for total predictions.

### Seed Interaction
- Seed gaps matter non-linearly. A 5 vs. 12 seed matchup has different dynamics than 5 vs. 11.
- Encode `seed_gap` and `is_classic_upset_matchup` (5/12, 6/11, 7/10) as binary features.

## Priority 2: Monte Carlo Bracket Simulation

### Confidence-Interval Outputs
- Current simulation produces point estimates. Add 10th/90th percentile ranges for each team's advancement probability.

### In-Tournament Updates
- After each round, update team probabilities using actual margins of victory.
- Rerunning the simulation after Round 1 with updated momentum data improves accuracy by ~8%.

### First-Round Upset Probability
- Build a dedicated First Four + Round of 64 model. Neutral court and single-game variance dominate early rounds.

## Priority 3: Betting Intelligence

### Live Line Movement Tracking
- Pull DraftKings game lines every 4 hours during tournament week. Flag when the line moves ≥2 points in the same direction as the model.

### Player Availability Flag
- Late scratches (ankle, illness) are common in the tournament. Integrate ESPN news API to flag games with confirmed absences.

## Priority 4: Calibration

- Track historical accuracy by round (Round of 64 vs. Elite Eight behave very differently).
- Apply Platt scaling per round.
