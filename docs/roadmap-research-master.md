# Research Dossier Implementation Roadmap — Master Index

*Derived from the Claude Opus 4.6 Bracket 2 Codex Research Dossier*

## Purpose

This document ties together four phase-specific roadmaps that translate the research dossier's 8 predictive theories and 20 improvement recommendations into concrete code changes for the march-madness prediction engine. Each phase is self-contained and can be implemented independently, though they build on each other in the recommended order.

---

## Phase Summary

| Phase | Focus | Files | New Features | Impact |
|-------|-------|-------|-------------|--------|
| [Phase 1](roadmap-research-phase1-core.md) | Core Engine | 5 modified | Logistic probability engine, composite ratings, AdjD boost | Moneyline log-loss −5–8 %, bracket accuracy +3–5 % |
| [Phase 2](roadmap-research-phase2-advanced.md) | Advanced Features | 3 modified, 2 new | Matchup matrix, 3PT variance, coaching, fatigue, travel | Moneyline accuracy +1–2 %, upset AUC +3–5 % |
| [Phase 3](roadmap-research-phase3-simulation.md) | Simulation | 2 modified | 10K+ Monte Carlo, sensitivity grid, champion profile, seed priors, scoring optimizer | Championship prob SE < 0.5 %, calibrated upset rates |
| [Phase 4](roadmap-research-phase4-live.md) | Live/Dynamic | 3 modified, 4 new | Daily updates, line movement, injury monitor, bracket loader | Spread MAE −0.3–0.6 pts, real-time adaptability |

---

## Mapping: Research Dossier → Implementation

### Theories

| # | Theory Name | Phase | Section | Rationale |
|---|------------|-------|---------|-----------|
| 1 | AdjD — The Silent Kingmaker | 1 | §1.3 | Elite defense correlates 3× with championships; easy binary feature |
| 2 | T-Rank Logistic Engine | 1 | §1.1 | Calibrated logistic function replaces arbitrary scale factor |
| 3 | Championship Profile Composite | 3 | §3.3 | Filters unrealistic champions using historical thresholds |
| 4 | Kill Shot — Style Collision Matrix | 2 | §2.1 | Matchup interactions capture style clashes missed by averages |
| 5 | Graph to Greatness (Multi-Source) | 1 | §1.2 | Fusing 3 rating sources smooths source-specific bias |
| 6 | VIG — Variance Is the Great Equalizer | 2 | §2.2 | 3PT variance is the primary upset driver |
| 7 | Historical Priors — Bayesian Anchor | 3 | §3.4 | Seed priors prevent model from straying from base rates |
| 8 | BPR — Bracket Pool Returns | 3 | §3.5 | Scoring optimization for pool strategy |

### Improvements

| # | Improvement | Phase | Section | Rationale |
|---|-----------|-------|---------|-----------|
| 1 | Selection Sunday Topology | 4 | §4.4 | Must ingest real bracket to make predictions actionable |
| 2 | Conference Tournament Results | 4 | §4.5 | Most recent form signal before NCAA tournament |
| 3 | Injury / Suspension Monitor | 4 | §4.3 | Single player absence shifts efficiency by 2–5 pts |
| 4 | Matchup-Specific Adjustments | 2 | §2.1 | Style collision drives upset variance |
| 5 | 3PT Variance Modeling | 2 | §2.2 | High variance = higher upset probability |
| 6 | Travel / Venue Effects | 2 | §2.5 | 0.8 pt per 500-mile disadvantage |
| 7 | Sensitivity Grid (28-Variant) | 3 | §3.2 | Identifies volatile picks before committing |
| 8 | Real-Time Line Movement | 4 | §4.2 | Sharp money reveals information model can't see |
| 9 | Conference Tournament Fatigue | 2 | §2.4 | 1.5 pt R1 underperformance for fatigued auto-bids |
| 10 | Coaching Experience Multiplier | 2 | §2.3 | 8 % win-rate edge for experienced coaches |
| 11 | Champion Win Shares | 3 | §3.3 | Only profile-matching teams realistically can win it all |
| 12 | Ensemble Rating Average | 1 | §1.2 | Already have 3 sources — just need to merge them |
| 13 | Monte Carlo 10K+ | 3 | §3.1 | Reduces championship prob SE from ±1.5 % to ±0.5 % |
| 14 | Logistic Probability Calibration | 1 | §1.1 | Scale factor 6.5 matches observed upset rates |
| 15 | Market Cross-Check | 1 | §1.1 | Logistic output vs market odds identifies edge |
| 16 | Bracket Scoring Optimization | 3 | §3.5 | Different pools need different strategies |
| 17 | Close Game Watchlist | 3 | §3.2 | Stability score flags close calls |
| 18 | Audit Trail | 3 | §3.2 | Sensitivity grid provides implicit audit |
| 19 | Automated Bracket Loader | 4 | §4.4 | Eliminates manual bracket setup |
| 20 | Dynamic Daily Updating | 4 | §4.1 | Fresh data = better predictions |

---

## Prioritization Rationale

**Phase 1 first** because it:
- Requires zero new data sources — everything needed is already in the codebase
- Fixes a known calibration issue (logistic scale factor too flat)
- Creates the composite rating that all subsequent phases depend on
- Has the highest expected accuracy improvement relative to effort

**Phase 2 second** because:
- Matchup features address the largest gap in the current feature set (style interactions)
- 3PT variance directly improves the upset detection module, which is a user-facing feature
- Coaching and fatigue data can be bootstrapped quickly for the 68 tournament teams

**Phase 3 third** because:
- Simulation improvements are multiplicative — they amplify the quality of Phases 1–2 features
- Sensitivity analysis provides a quality-control layer over all predictions
- Champion profiling refines the most high-stakes prediction (tournament winner)

**Phase 4 last** because:
- Dynamic features require infrastructure (scheduled jobs, data pipelines) on top of the statistical improvements
- Line movement tracking needs ongoing Odds API access and storage
- Injury monitoring requires manual data entry or a scraping pipeline
- These are the most operationally complex but least statistically impactful per change

---

## Current Model Baseline (for comparison after implementation)

| Metric | Current Value | Source |
|--------|---------------|--------|
| Moneyline ensemble accuracy | ~65–72 % | `model_training.py` output |
| Spread MAE | ~8–10 points | XGBoost + RF + LR ensemble |
| Total MAE | ~10–12 points | Same ensemble |
| Bracket simulation runs | 1,000 | `bracket_simulation.py` default |
| Features per game | 11 (efficiency diffs + KenPom + BartTorvik) | `predictions.py` |
| Rating sources used | 3 (separate, not merged) | KenPom + BartTorvik + Haslametrics |
| Logistic scale factor | 15 (fallback) | `features.py` line ~268 |

### Expected Post-Implementation Metrics

| Metric | Expected Value | Phase |
|--------|---------------|-------|
| Moneyline accuracy | 70–76 % | Phases 1–2 |
| Spread MAE | 7–9 points | Phases 1–2–4 |
| Total MAE | 9–11 points | Phase 2 |
| Simulation runs | 10,000+ | Phase 3 |
| Features per game | 25–30 | Phases 1–2 |
| Championship prob SE | < 0.5 % | Phase 3 |
| Logistic scale factor | 6.5 | Phase 1 |

---

## Quick-Start Checklist

- [ ] **Phase 1.1**: Update logistic scale factor from 15 → 6.5 in `features.py`
- [ ] **Phase 1.2**: Add `build_composite_ratings()` to `efficiency_loader.py`
- [ ] **Phase 1.3**: Add `apply_adjd_championship_boost()` to `features.py`
- [ ] **Phase 2.1**: Add `calculate_matchup_features()` to `features.py`
- [ ] **Phase 2.2**: Add `calculate_three_pt_variance_features()` to `features.py`
- [ ] **Phase 3.1**: Increase default `num_simulations` to 10,000
- [ ] **Phase 3.4**: Add `HISTORICAL_SEED_WIN_RATES` and `blend_with_prior()`
- [ ] **Phase 4.1**: Enhance GitHub Actions schedule for tournament weeks
