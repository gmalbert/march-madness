# Roadmap: Live Prediction Pipeline & Betting Automation

> **Priority**: P3 — High value but depends on model improvements being in place  
> **Estimated Effort**: Medium-High  
> **Not in sports-quant analysis** — identified from reviewing current betting infrastructure  
> **Impact**: Turn predictions into actionable, timely betting decisions

---

## Problem Statement

The current repo has pieces of a live betting pipeline (live odds fetching, opening line tracking, value bet detection, Kelly criterion sizing, ROI tracking), but they're not connected into an automated workflow. During March Madness, the window between bracket release and first games is short — automation saves time and ensures consistency.

---

## Part 1: Automated Prediction Pipeline

### Trigger Points

| Event | Action | Timing |
|-------|--------|--------|
| Selection Sunday | Generate full bracket predictions | Immediately after bracket reveal |
| Each tournament day | Refresh predictions with live odds | Morning of game day |
| Line movement detected | Re-evaluate value bets | Continuous |
| Game result finalized | Update ROI tracker | After each game |

### Pipeline Design

```
Selection Sunday:
  1. Download fresh KenPom + BartTorvik (pre-tournament snapshots)
  2. Load trained models
  3. Generate bracket predictions (deterministic + MC)
  4. Compute all pairwise probabilities
  5. Export bracket to Streamlit dashboard
  6. Generate survivor pool picks

Game Day Morning:
  1. Fetch live odds (spreads, totals, moneylines)
  2. Compare model probabilities to implied odds
  3. Identify value bets (model edge > threshold)
  4. Apply Kelly criterion for bet sizing
  5. Push picks to dashboard / export
  
Post-Game:
  1. Record actual results
  2. Update ROI tracker
  3. Adjust for bracket cascade (forwarded winners)
```

---

## Part 2: Value Bet Detection Refinements

### Current
`underdog_value.py` has `find_value_bets()` and `calculate_kelly_criterion()`.

### Improvements

**Confidence-gated betting** (from sports-quant NFL model):
Don't bet every game — only bet when the model has a clear edge:

```python
def should_bet(model_prob, implied_prob, min_edge=0.05, max_vig=0.10):
    """Only bet when model edge exceeds minimum threshold."""
    edge = model_prob - implied_prob
    if edge < min_edge:
        return False
    # Also check that we're not paying too much vig
    vig = (1 / implied_prob_home + 1 / implied_prob_away) - 1
    if vig > max_vig:
        return False
    return True
```

**Multi-model consensus** (from sports-quant NFL approach):
Only bet when multiple models agree:
```python
# Train 50 models, require top 3 to agree on the same side
consensus = all(model.predict(game) == "OVER" for model in top_3_models)
```

**Algorithm score** (sports-quant's key innovation for NFL):
Weighted blend of per-model confidence-bin accuracy — games in higher algorithm-score bins are historically more reliable.

---

## Part 3: Line Movement Integration

### Current
We have `opening_line_database.py` and line movement tracking in `scripts/odds_api_integration.py`.

### Enhancement: Sharp Money Detection

```python
def detect_sharp_money(opening_line, current_line, public_betting_pct):
    """Detect reverse line movement (line moves against public money)."""
    line_move = current_line - opening_line
    if public_betting_pct > 60 and line_move < 0:
        # Public on one side but line moves the other way
        # → Sharp money on the opposite side
        return "sharp_contra"
    return None
```

### Line Value Decay
Opening lines are the most +EV (before market efficiency kicks in). Track when lines were captured and discount edge for stale lines.

---

## Part 4: ROI Dashboard

### Current
`roi_tracker.py` tracks basic ROI in JSON.

### Enhancement: Comprehensive Analytics

| Metric | Description |
|--------|-------------|
| Overall ROI | Total profit / total wagered |
| ROI by bet type | Spread, total, moneyline separately |
| ROI by confidence bin | How profitable are high-confidence bets vs low? |
| Closing line value (CLV) | Did our bets have positive CLV? (gold standard) |
| Drawdown tracking | Maximum loss from peak |
| Bankroll simulation | Monte Carlo of bankroll trajectory |

### CLV (Closing Line Value)

The most important metric for long-term profitability:
```python
def calculate_clv(bet_line, closing_line):
    """Positive CLV = you got a better number than the market."""
    return bet_line - closing_line  # For spreads
```

If you consistently beat the closing line, you will be profitable long-term.

---

## Part 5: Notification System

When value bets are identified, send alerts:

| Channel | Priority |
|---------|----------|
| Streamlit dashboard | Primary — always visible |
| JSON/CSV export | For programmatic consumption |
| Discord webhook (optional) | Real-time alerts |

---

## Acceptance Criteria

- [ ] Automated prediction pipeline triggered by config/schedule
- [ ] Value bet detection with confidence gating and minimum edge
- [ ] Multi-model consensus requirement for bet recommendations
- [ ] Line movement tracking integrated into value assessment
- [ ] ROI dashboard with CLV, drawdown, and per-category breakdowns
- [ ] Selection Sunday → game day → post-game workflow documented and tested

---

## Dependencies

- Calibrated probabilities (critical — uncalibrated probs = bad value bets)
- Live odds API already integrated
- Model versioning (to track which model version generated which bets)
