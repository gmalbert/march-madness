# Roadmap: Data Leakage Prevention & Pre-Tournament Snapshots

> **Priority**: P0 — Foundational integrity issue  
> **Estimated Effort**: Low-Medium  
> **Source**: sports-quant scraper design, model-analysis-findings.md  
> **Impact**: Eliminates contaminated training data; results become trustworthy

---

## Problem Statement

Our current data pipeline has a **critical data leakage issue**: we scrape BartTorvik season-end CSVs that **include tournament game results**. When we train on 2022 data and predict the 2022 tournament, the BartTorvik stats already reflect how teams performed IN the tournament. This makes our backtest accuracy artificially inflated and untrustworthy.

Sports-quant solves this by using BartTorvik's "Time Machine" API to pull **pre-tournament snapshots** — ratings as they existed on Selection Sunday, before any tournament games.

---

## Part 1: BartTorvik Time Machine API

### Current Approach (Broken)
```python
# download_barttorvik.py — Selenium scrapes season-end CSV
# This CSV includes tournament game results → data leakage
```

### Fixed Approach
```python
import urllib.request, json

def download_barttorvik_snapshot(year: int, date: str = None) -> pd.DataFrame:
    """Download pre-tournament BartTorvik ratings via Time Machine API.
    
    No Selenium required. Returns ratings as of the given date.
    """
    if date is None:
        # Selection Sunday dates — ratings before tournament begins
        selection_sundays = {
            2016: "20160313", 2017: "20170312", 2018: "20180311",
            2019: "20190317", 2021: "20210314", 2022: "20220313",
            2023: "20230312", 2024: "20240317", 2025: "20250316",
            2026: "20260315",
        }
        date = selection_sundays.get(year, f"{year}0315")
    
    url = f"https://barttorvik.com/getadvstats.php?year={year}&date={date}"
    # ... parse JSON response into DataFrame
```

### BartTorvik Time Machine Fields Available

```
team, conf, record, adjoe, adjde, barthag, adjt,
sos, ncsos, elite_sos, wab, qual_o, qual_d, qual_barthag
```

This gives us 13 stat columns per team — significantly more than our current 2 (AdjOE, AdjDE).

---

## Part 2: KenPom Snapshot Integrity

### Current Approach
Our `download_kenpom.py` scrapes the current KenPom page. This is fine for the current season but may not give us clean historical data.

### Recommended Fix
- For the current season: continue scraping live KenPom (it shows current-year ratings)
- For historical seasons: use the Wayback Machine (Internet Archive) or store annual snapshots
- Sports-quant scrapes via `barttorvik.com/kenpom.php?y=YYYY` for historical KenPom (which mirrors KenPom data by year)

### Key Principle
**All features must represent information available BEFORE the tournament starts.** Any stat that could be influenced by tournament outcomes is leaking.

---

## Part 3: Training Data Temporal Integrity

Beyond scraping, enforce temporal integrity in the training pipeline:

### Checklist

- [ ] **BartTorvik**: Use pre-Selection Sunday snapshots only
- [ ] **KenPom**: Use pre-tournament ratings (end of regular season / conference tournaments)
- [ ] **NET Rankings**: Use pre-tournament release (these stop updating once the bracket is set)
- [ ] **Win-loss records**: Exclude tournament games
- [ ] **Efficiency metrics**: Based on regular season + conference tournament only
- [ ] **Betting lines**: Opening lines only (no live/closing lines that react to sharp money)

### Validation

Add an automated check that no training feature could be influenced by tournament outcomes:

```python
def validate_temporal_integrity(training_df, tournament_start_dates):
    """Verify no training features include tournament-period data."""
    for year in training_df["year"].unique():
        tourney_start = tournament_start_dates[year]
        year_data = training_df[training_df["year"] == year]
        # Check that all stat columns reflect pre-tournament state
        # Flag any suspicious values
```

---

## Part 4: Opening Line Database Integrity

Our `opening_lines.json` stores betting lines. Ensure these are truly **opening** lines (set before game day) and not closing lines (which incorporate injury news, sharp action, etc.).

Sports-quant doesn't use betting lines as features, but if we do, we need:
- Clear provenance: when was this line captured?
- Opening vs closing distinction
- No in-game or live line data in training features

---

## Acceptance Criteria

- [ ] BartTorvik data sourced from Time Machine API (pre-tournament snapshots)
- [ ] No Selenium dependency for BartTorvik (simpler, more reliable)
- [ ] KenPom historical data uses pre-tournament snapshots
- [ ] Automated validation that no training features leak tournament outcomes
- [ ] Selection Sunday dates mapped for all historical years
- [ ] Documentation of what "pre-tournament" means for each data source

---

## Dependencies

- None — this should be done first since it affects all downstream analysis

---

## Risks

| Risk | Severity | Mitigation |
|------|----------|------------|
| Time Machine API may change or rate-limit | Low | Cache responses; it's a simple JSON endpoint |
| Historical snapshot dates may be imprecise | Low | Use day before Selection Sunday; a few days won't matter |
| Existing backtest results become invalid | Expected | This is the point — current results are overfit. Honest numbers are more useful |
