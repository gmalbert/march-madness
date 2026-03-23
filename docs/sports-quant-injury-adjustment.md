# Sports-Quant Injury Adjustment System

> Source: [thadhutch/sports-quant](https://github.com/thadhutch/sports-quant) — `docs/injury-adjustment-spec.md`  
> Purpose: Document the injury adjustment system for integration into our march-madness project. This is a "Phase 5" feature — powerful but requires significant new data pipelines.

---

## Executive Summary

KenPom and BartTorvik ratings are **season-long aggregates** that don't reflect late-season player absences. When a star player tears an ACL in the conference tournament, the model still predicts based on full-strength ratings — overvaluing the injured team.

The injury adjustment system:
1. Scrapes player-level stats from Sports Reference
2. Computes each player's importance to their team's efficiency
3. Fetches injury reports from ESPN
4. Adjusts team-level stats (AdjO, AdjD) based on absent players
5. Feeds adjusted stats through the existing pipeline (no model changes needed)

---

## Architecture

```
ESPN Injury Page  +  Sports-Reference Player Stats
       │                        │
       ▼                        ▼
  Injury Reports          Player Importance
  (player, status)         Scoring (0-1)
       │                        │
       ▼                        ▼
       ┌────────────────────────┘
       │  Adjustment Calculator
       │  adjusted_AdjO = AdjO * (1 - lost_frac * degradation)
       │  adjusted_AdjD = AdjD * (1 + lost_frac * degradation)
       ▼
  Adjusted KenPom Stats
  (drop-in replacement in FeatureLookup)
       │
       ├──→ predict.py (unchanged)
       ├──→ simulate.py (unchanged)  
       └──→ survivor.py (unchanged)
```

**Key design principle:** The injury system adjusts stats **before** they enter the model pipeline. The model itself is unchanged — it still sees the same feature vectors, just with more accurate numbers.

---

## Data Models

```python
# injury_models.py — Data structures for injury adjustment

from dataclasses import dataclass
from enum import Enum


class InjuryStatus(Enum):
    """Player availability status from injury reports."""
    OUT = "out"                  # Will not play
    DOUBTFUL = "doubtful"        # Very unlikely
    QUESTIONABLE = "questionable"  # Coin flip
    PROBABLE = "probable"        # Expected to play
    HEALTHY = "healthy"          # No issue


@dataclass(frozen=True)
class PlayerStats:
    """Season-level statistics for a single player."""
    player_name: str
    team: str                    # Standardized team name
    year: int
    games: int
    minutes_per_game: float
    minutes_pct: float           # Fraction of team's total minutes
    usage_rate: float            # USG% — fraction of possessions used
    offensive_rating: float      # ORtg
    defensive_rating: float      # DRtg
    box_plus_minus: float        # BPM — overall impact
    offensive_bpm: float         # OBPM
    defensive_bpm: float         # DBPM
    points_per_game: float
    rebounds_per_game: float
    assists_per_game: float


@dataclass(frozen=True)
class InjuryReport:
    """Injury status for a single player."""
    player_name: str
    team: str
    status: InjuryStatus
    injury_description: str
    report_date: str             # ISO format: "2025-03-15"
    source: str                  # "espn", "manual", "llm"


@dataclass(frozen=True)
class PlayerImpact:
    """Quantified contribution of a player to their team's efficiency."""
    player_name: str
    team: str
    year: int
    importance_score: float      # 0.0 to 1.0, sums to 1.0 across roster
    adj_o_contribution: float    # Offensive importance (0-1)
    adj_d_contribution: float    # Defensive importance (0-1)


@dataclass(frozen=True)
class AdjustedTeamStats:
    """Team stats after applying injury adjustments."""
    team: str
    year: int
    original_stats: dict         # Original KenPom values
    adjusted_stats: dict         # Adjusted KenPom values
    adjustments_applied: tuple   # Which players caused adjustments
    adjustment_source: str       # Description of trigger
```

---

## Player Importance Scoring

Three-signal composite score:

| Signal | Weight | Rationale |
|--------|--------|-----------|
| Minutes share | 0.35 | More minutes = more impact |
| Usage-weighted minutes | 0.30 | High usage + high minutes = drives offense |
| BPM contribution | 0.35 | Best single stat for overall impact |

```python
# player_importance.py — Score player contributions

import numpy as np


def score_player_importance(players: list[PlayerStats]) -> list[PlayerImpact]:
    """Score each player's contribution to team efficiency.
    
    Algorithm:
        For each player:
            minutes_component = player.minutes_pct
            usage_component = (player.usage_rate / 100) * player.minutes_pct
            bpm_component = player.box_plus_minus * player.minutes_pct
        
        Normalize each component across the roster so they sum to 1.0.
        Final score = 0.35 * minutes_norm + 0.30 * usage_norm + 0.35 * bpm_norm
        
        For offensive/defensive split:
            adj_o_contribution uses OBPM instead of BPM
            adj_d_contribution uses DBPM instead of BPM
    
    Edge cases:
        - Negative BPM: shift all values by abs(min) + 0.1 before computing
        - Single dominant player: cap at 0.60, redistribute excess
        - Players with < 5 games: excluded from scoring
    """
    # Filter to players with enough games
    eligible = [p for p in players if p.games >= 5]
    
    if not eligible:
        return []
    
    # Raw components
    minutes = np.array([p.minutes_pct for p in eligible])
    usage_weighted = np.array([
        (p.usage_rate / 100) * p.minutes_pct for p in eligible
    ])
    
    # BPM: shift to all-positive
    bpm_values = np.array([p.box_plus_minus for p in eligible])
    bpm_shifted = bpm_values - bpm_values.min() + 0.1
    bpm_weighted = bpm_shifted * minutes
    
    obpm_values = np.array([p.offensive_bpm for p in eligible])
    obpm_shifted = obpm_values - obpm_values.min() + 0.1
    obpm_weighted = obpm_shifted * minutes
    
    dbpm_values = np.array([p.defensive_bpm for p in eligible])
    dbpm_shifted = dbpm_values - dbpm_values.min() + 0.1
    dbpm_weighted = dbpm_shifted * minutes
    
    # Normalize each component to sum to 1.0
    def normalize(arr):
        total = arr.sum()
        return arr / total if total > 0 else np.ones_like(arr) / len(arr)
    
    min_norm = normalize(minutes)
    usage_norm = normalize(usage_weighted)
    bpm_norm = normalize(bpm_weighted)
    obpm_norm = normalize(obpm_weighted)
    dbpm_norm = normalize(dbpm_weighted)
    
    # Composite scores
    overall = 0.35 * min_norm + 0.30 * usage_norm + 0.35 * bpm_norm
    offensive = 0.35 * min_norm + 0.30 * usage_norm + 0.35 * obpm_norm
    defensive = 0.35 * min_norm + 0.30 * usage_norm + 0.35 * dbpm_norm
    
    # Cap dominant players at 0.60
    MAX_IMPORTANCE = 0.60
    for arr in [overall, offensive, defensive]:
        excess_mask = arr > MAX_IMPORTANCE
        if excess_mask.any():
            excess = (arr[excess_mask] - MAX_IMPORTANCE).sum()
            arr[excess_mask] = MAX_IMPORTANCE
            # Redistribute to non-capped players
            non_capped = ~excess_mask
            if non_capped.any():
                arr[non_capped] += excess * (arr[non_capped] / arr[non_capped].sum())
    
    impacts = []
    for i, player in enumerate(eligible):
        impacts.append(PlayerImpact(
            player_name=player.player_name,
            team=player.team,
            year=player.year,
            importance_score=float(overall[i]),
            adj_o_contribution=float(offensive[i]),
            adj_d_contribution=float(defensive[i]),
        ))
    
    return impacts
```

---

## Injury Adjustment Calculation

```python
# injury_adjustment.py — Compute adjusted team stats

# Status multipliers: how much of the absence to count
STATUS_MULTIPLIER = {
    InjuryStatus.OUT: 1.0,           # Definitely not playing
    InjuryStatus.DOUBTFUL: 0.75,     # Very unlikely
    InjuryStatus.QUESTIONABLE: 0.25, # Coin flip — discount heavily
    InjuryStatus.PROBABLE: 0.05,     # Almost certainly playing
    InjuryStatus.HEALTHY: 0.0,       # No adjustment
}

# How much of lost production translates to team degradation
# 0.5 = losing 30% of production degrades the team by 15%
# Based on NBA research: replacement players absorb 40-60% of production
DEFAULT_DEGRADATION_FACTOR = 0.5


def compute_adjustments(
    team: str,
    year: int,
    original_stats: dict,
    player_impacts: list[PlayerImpact],
    injuries: list[InjuryReport],
    degradation_factor: float = DEFAULT_DEGRADATION_FACTOR,
) -> AdjustedTeamStats:
    """Compute adjusted KenPom stats based on player absences.
    
    Algorithm:
        1. For each injured player with status != HEALTHY:
           - Look up their PlayerImpact
           - lost_o = adj_o_contribution * status_multiplier * degradation_factor
           - lost_d = adj_d_contribution * status_multiplier * degradation_factor
        
        2. Sum lost_o and lost_d across all injured players
        
        3. Apply to stats:
           - adjusted_AdjO = original_AdjO * (1 - total_lost_o)
           - adjusted_AdjD = original_AdjD * (1 + total_lost_d)
             (defense gets WORSE = higher number)
           - adjusted_AdjEM = adjusted_AdjO - adjusted_AdjD
    
    Stats NOT adjusted:
        - AdjT: Tempo is a coaching scheme, not single-player dependent
        - SOS/NCSOS: Schedule is fixed
        - Rank columns: Can't recompute without adjusting all teams
    """
    # Build impact lookup
    impact_by_player = {pi.player_name: pi for pi in player_impacts}
    
    total_lost_o = 0.0
    total_lost_d = 0.0
    adjustments_applied = []
    
    for injury in injuries:
        if injury.status == InjuryStatus.HEALTHY:
            continue
        
        impact = impact_by_player.get(injury.player_name)
        if impact is None:
            continue  # Player not found in roster
        
        multiplier = STATUS_MULTIPLIER[injury.status]
        lost_o = impact.adj_o_contribution * multiplier * degradation_factor
        lost_d = impact.adj_d_contribution * multiplier * degradation_factor
        
        total_lost_o += lost_o
        total_lost_d += lost_d
        adjustments_applied.append(impact)
    
    # Cap maximum degradation at 25%
    total_lost_o = min(total_lost_o, 0.25)
    total_lost_d = min(total_lost_d, 0.25)
    
    if total_lost_o == 0 and total_lost_d == 0:
        return AdjustedTeamStats(
            team=team, year=year,
            original_stats=original_stats,
            adjusted_stats=dict(original_stats),
            adjustments_applied=tuple(),
            adjustment_source="no injured players",
        )
    
    adjusted = dict(original_stats)
    
    # Adjust offensive efficiency (lower = worse)
    if "AdjO" in adjusted:
        adjusted["AdjO"] = original_stats["AdjO"] * (1 - total_lost_o)
    if "ORtg" in adjusted:
        adjusted["ORtg"] = original_stats["ORtg"] * (1 - total_lost_o)
    
    # Adjust defensive efficiency (higher = worse for defense)
    if "AdjD" in adjusted:
        adjusted["AdjD"] = original_stats["AdjD"] * (1 + total_lost_d)
    if "DRtg" in adjusted:
        adjusted["DRtg"] = original_stats["DRtg"] * (1 + total_lost_d)
    
    # Recompute net efficiency
    if "AdjEM" in adjusted and "AdjO" in adjusted and "AdjD" in adjusted:
        adjusted["AdjEM"] = adjusted["AdjO"] - adjusted["AdjD"]
    if "NetRtg" in adjusted and "ORtg" in adjusted and "DRtg" in adjusted:
        adjusted["NetRtg"] = adjusted["ORtg"] - adjusted["DRtg"]
    
    return AdjustedTeamStats(
        team=team, year=year,
        original_stats=original_stats,
        adjusted_stats=adjusted,
        adjustments_applied=tuple(adjustments_applied),
        adjustment_source=f"injuries: {[i.player_name for i in injuries if i.status != InjuryStatus.HEALTHY]}",
    )
```

---

## Pipeline Integration

The adjustment system plugs into `FeatureLookup` with a single optional parameter:

```python
# In feature_lookup.py:
class FeatureLookup:
    def __init__(
        self,
        kenpom_df: pd.DataFrame,
        adjustments: dict[tuple[str, int], dict[str, float]] | None = None,
    ):
        """
        adjustments: Optional dict of (team_name, year) -> adjusted stat values.
            If provided, these values override KenPom stats for specified teams.
            Other teams are unaffected.
        """
        # ... existing init code ...
        self._adjustments = adjustments or {}
    
    def get_team(self, team: str, year: int) -> dict:
        stats = self._base_index[(team, year)]
        # Overlay adjustments if any exist for this team
        adj = self._adjustments.get((team, year))
        if adj:
            stats = {**stats, **adj}
        return stats
```

**Backwards compatible:** Existing code that doesn't pass `adjustments` gets identical behavior.

### Orchestrator

```python
def build_injury_adjustments(
    year: int,
    tournament_teams: list[str] | None = None,
    injury_overrides: list[InjuryReport] | None = None,
    degradation_factor: float = DEFAULT_DEGRADATION_FACTOR,
) -> dict[tuple[str, int], dict[str, float]]:
    """Build injury adjustments for all tournament teams.
    
    Steps:
        1. Load player stats (from CSV or scrape if missing)
        2. Load injury reports (from ESPN or CSV)
        3. Merge any manual overrides
        4. Compute importance scores
        5. Compute adjustments for teams with injured players
        6. Return adjustments dict in FeatureLookup format
    
    Returns:
        Dict of (team_name, year) -> adjusted_stats_dict.
        Only includes teams that have injured players.
    """
    # Load player stats
    player_stats = load_player_stats(year, tournament_teams)
    
    # Load injury reports
    injuries = scrape_espn_injuries()
    if injury_overrides:
        injuries.extend(injury_overrides)
    
    # Group by team
    stats_by_team = {}
    for ps in player_stats:
        stats_by_team.setdefault(ps.team, []).append(ps)
    
    injuries_by_team = {}
    for inj in injuries:
        injuries_by_team.setdefault(inj.team, []).append(inj)
    
    adjustments = {}
    for team, team_injuries in injuries_by_team.items():
        if team not in stats_by_team:
            continue
        
        impacts = score_player_importance(stats_by_team[team])
        
        # Get original stats (would come from KenPom data)
        original = get_team_stats(team, year)
        
        result = compute_adjustments(
            team, year, original, impacts,
            team_injuries, degradation_factor,
        )
        
        if result.adjustments_applied:
            adjustments[(team, year)] = result.adjusted_stats
    
    return adjustments


# Usage in prediction pipeline:
# adjustments = build_injury_adjustments(year=2025)
# feature_lookup = FeatureLookup(kenpom_df, adjustments=adjustments)
# result = simulate_bracket(year, models, feature_lookup, ...)
```

---

## LLM-Powered Injury Parsing (Advanced)

Parse unstructured text (tweets, press conferences) into structured injury reports:

```python
# llm_injury_parser.py — Claude API for injury text extraction

def parse_injury_text(
    raw_text: str,
    known_teams: list[str] | None = None,
) -> list[InjuryReport]:
    """Parse unstructured text into structured injury reports.
    
    Uses Claude API with a structured extraction prompt.
    Validates extracted player names against scraped rosters.
    Rejects extractions where the player cannot be fuzzy-matched.
    
    Validation pipeline:
        1. Claude extracts raw JSON from text
        2. Parse JSON into candidate InjuryReport objects
        3. For each candidate:
           a. Fuzzy-match team against known team names
           b. Fuzzy-match player against that team's roster
           c. If either match fails, discard + log warning
        4. Return only validated reports
    """
    # System prompt for extraction
    SYSTEM_PROMPT = """You are a sports data extraction tool. Given unstructured text about
college basketball player injuries, extract structured injury information.

For each injury mentioned, extract:
- player_name: Full name of the player
- team: College team name
- status: One of "out", "doubtful", "questionable", "probable"
- injury_description: Brief description of the injury

Return a JSON array. If no injuries are mentioned, return [].
Do not infer or guess — only extract what is explicitly stated."""
    
    # Call Claude API (requires anthropic package)
    # response = client.messages.create(
    #     model="claude-sonnet-4-20250514",
    #     system=SYSTEM_PROMPT,
    #     messages=[{"role": "user", "content": raw_text}],
    # )
    # 
    # Parse response JSON and validate against rosters
    # ...
    pass
```

---

## Implementation Phases

| Phase | Deliverable | Effort |
|-------|------------|--------|
| 1 | Data models + player stats scraper | Medium |
| 2 | Player importance scoring | Low |
| 3 | ESPN injury scraper + adjustment calculation | Medium |
| 4 | FeatureLookup integration | Low |
| 5 | LLM-powered injury parsing | Medium |
| 6 | Calibration + backtesting | Medium |

### Phase 1 Dependencies
- Sports Reference scraping (68 teams × 1 request each = manageable)
- Team name mapping between Sports Reference, KenPom, and ESPN

### Risks & Mitigations
| Risk | Mitigation |
|------|-----------|
| Sports Reference blocks scraping | Rate limit (3s delays), cache, fall back to BartTorvik |
| Team name mismatches | Dedicated name mapping, fail loudly on unmatched |
| Degradation factor poorly calibrated | Conservative default (0.5), Phase 6 calibration |
| Sparse college injury reporting | Focus on high-impact absences (starters, >25% minutes) |
| LLM hallucinations | Validate all extractions against roster, reject unmatched |

---

## Priority Assessment

This is a **medium-priority, high-complexity** feature. It adds genuine predictive value (tournament upsets frequently correlate with key absences), but requires:
- Two new scrapers (Sports Reference + ESPN)
- Complex name matching across 3+ data sources
- Careful calibration of the degradation factor

**Recommended approach:** Implement after the higher-ROI items (difference features, BartTorvik Time Machine, calibration, tuning) are in place. The injury system works best when the base model is already strong.
