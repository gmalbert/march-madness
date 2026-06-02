# March Madness Predictor — Next 5 Features to Implement

> **Based on:** Codebase gap analysis as of July 2025

---

## Feature 1: Interactive Bracket Visualizer

**Why:** The `docs/BRACKET_SIMULATOR.md` documents this as a planned feature but it has not been built. A Plotly/SVG interactive bracket is the single most visible missing feature — bracket visualization is what casual users come for during tournament time.

**How:**
1. Build a Plotly custom layout using shapes/annotations to render the 68-team bracket (64 after play-in)
2. Drive each slot with Monte Carlo simulation win probabilities (already computed in the model)
3. Use team seed and name as axis labels; color-code nodes by predicted win probability
4. Add a "Simulate" button that re-runs 10,000 trials and updates the bracket visualization
5. Allow users to lock specific bracket picks and see how it changes overall accuracy

**Complexity:** Medium

---

## Feature 2: Seed Matchup Historical Database

**Why:** Historical 1 vs 16, 5 vs 12, 8 vs 9 ATS records are the most commonly cited NCAA tournament statistics. These seed-matchup priors are a meaningful calibration feature — "12 seeds cover against 5 seeds 50% of the time historically" should be captured as a model feature.

**How:**
1. Compile 1985–present seed matchup ATS records from public NCAA tournament result databases
2. Store in `data_files/seed_matchup_history.csv`: `higher_seed`, `lower_seed`, `ats_cover_pct`, `avg_point_diff`, `upset_pct`
3. Add `seed_matchup_ats_prior` as a model feature (Bayesian prior over the model probability)
4. Display on each bracket match card: "Historical ATS: 12 seeds 50% vs 5 seeds (last 20 tournaments)"

**Complexity:** Low

---

## Feature 3: Live Line Movement Tracker

**Why:** Tournament lines open weeks in advance. Opening vs current spread snapshots reveal which games have seen sharp action. This is documented in the roadmap but not yet implemented.

**How:**
1. Add `data_files/raw/march_madness_odds_snapshots.csv` with columns: `game_id`, `snapshot_time`, `home_spread`, `away_spread`, `total`
2. Store opening lines when the bracket is announced + daily snapshots via the existing `fetch_odds.py` pattern
3. Add `pages/line_movement.py` with a per-game chart of line movement from open to current
4. Compute "line move direction" alignment with model pick (confirming signal vs contrarian signal)

**Complexity:** Low

---

## Feature 4: Bankroll Persistence Across Tournaments

**Why:** The Kelly staking logic generates bet recommendations but there is no cross-tournament persistence of outcomes. Building a multi-year P&L tracker would prove the model's long-term value and identify seasonal drift.

**How:**
1. Create `data_files/bankroll_history.json` persisting each tournament's bet outcomes: year, game_id, pick, odds, stake, outcome, P&L
2. Update after each game resolves using the same result-reconciliation pattern as the picks system
3. Add `pages/performance.py` showing: per-tournament ROI, cumulative P&L chart, accuracy by round (Round of 64 vs Elite Eight)
4. Include a calibration chart: model probability decile vs actual win rate across all tournament history

**Complexity:** Medium

---

## Feature 5: Upset Probability Heatmap by Bracket Region

**Why:** Certain bracket regions have historically been more volatile (more upsets) than others due to seeding luck, geographic travel factors, or scheduling. A "volatility heatmap" per bracket region showing predicted upset probability would be a unique visualization.

**How:**
1. For each of the 4 bracket regions (South, East, West, Midwest), compute the average upset probability per round using the Monte Carlo simulation
2. Render as a Plotly heatmap: region × round with color scale showing upset probability
3. Add overlay: how each region's historical upset rate (1985–present) compares to the model's prediction
4. Display prominently on the home page during selection weekend

**Complexity:** Medium
