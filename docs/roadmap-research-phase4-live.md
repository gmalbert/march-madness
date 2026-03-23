# Phase 4: Live & Dynamic Systems

*Based on the Claude Opus 4.6 Bracket Codex Research Dossier — Improvements #1–3, #8, #19–20*

## Overview

Phase 4 introduces time-sensitive, dynamic capabilities: real-time line movement tracking, injury monitoring, automatic daily rating updates, and Selection Sunday topology refresh. These features transform the system from a static pre-tournament tool into a living prediction engine that adapts throughout March.

---

## 4.1 Dynamic Daily Rating Updates

### Why

Improvement #20 ("Dynamic Updating") notes that efficiency ratings change daily as games are played. The current system downloads ratings once via GitHub Actions (`update-efficiency-ratings.yml`) and uses them for the entire tournament. During conference tournaments (early-to-mid March) and between NCAA tournament rounds, teams' metrics can shift meaningfully — a key injury in the conference tournament final, a hot shooting streak, or a tempo change down the stretch.

### Expected Effect

- **Spread model**: Fresh ratings reduce stale-data error. Historical analysis shows 0.3–0.6 point MAE reduction when ratings are updated within 24 hours of tip-off vs. 5+ days stale.
- **Bracket simulation**: Re-running Monte Carlo with updated ratings after each round produces more accurate Sweet 16, Elite 8, and Final Four probabilities.
- **User experience**: The Streamlit app shows "Last Updated" timestamps, building user trust.

### Implementation

Enhance the GitHub Actions workflow:

```yaml
# .github/workflows/update-efficiency-ratings.yml
name: Update Efficiency Ratings

on:
  schedule:
    # During tournament weeks (mid-March to early April), run twice daily
    - cron: '0 8,20 * 3-4 *'    # 8 AM and 8 PM UTC in March-April
    # Rest of year, weekly
    - cron: '0 9 * 1-2,5-12 1'  # Monday 9 AM UTC
  workflow_dispatch:              # Manual trigger

jobs:
  update-ratings:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: pip install -r requirements-actions.txt

      - name: Download KenPom
        run: python download_kenpom.py
        continue-on-error: true

      - name: Download BartTorvik
        run: python download_barttorvik.py
        continue-on-error: true

      - name: Download Haslametrics
        run: python download_haslametrics.py
        continue-on-error: true

      - name: Build composite ratings
        run: python -c "
          from data_tools.efficiency_loader import EfficiencyDataLoader
          loader = EfficiencyDataLoader()
          composite = loader.build_composite_ratings()
          composite.to_csv('data_files/composite_ratings.csv', index=False)
          print(f'Composite ratings updated: {len(composite)} teams')
        "

      - name: Record update timestamp
        run: |
          echo "{\"last_updated\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\", \"source\": \"github_actions\"}" > data_files/ratings_metadata.json

      - name: Commit and push
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data_files/
          git diff --staged --quiet || git commit -m "Auto-update efficiency ratings $(date -u +%Y-%m-%d)"
          git push
```

Add a staleness check to `predictions.py`:

```python
import json
from datetime import datetime, timezone

def check_data_freshness() -> dict:
    """Check how old the efficiency ratings are."""
    meta_path = Path('data_files/ratings_metadata.json')
    if not meta_path.exists():
        return {'fresh': False, 'age_hours': None, 'message': 'No metadata found'}

    with open(meta_path) as f:
        meta = json.load(f)

    last_updated = datetime.fromisoformat(meta['last_updated'].replace('Z', '+00:00'))
    age = datetime.now(timezone.utc) - last_updated
    age_hours = age.total_seconds() / 3600

    return {
        'fresh': age_hours < 24,
        'age_hours': round(age_hours, 1),
        'last_updated': meta['last_updated'],
        'message': f"Ratings updated {round(age_hours, 1)}h ago"
            if age_hours < 48
            else f"⚠️ Ratings are {round(age_hours / 24, 1)} days old — consider refreshing",
    }
```

Display in the Streamlit sidebar:

```python
# In predictions.py sidebar
freshness = check_data_freshness()
if freshness['fresh']:
    st.sidebar.success(freshness['message'])
else:
    st.sidebar.warning(freshness['message'])
```

---

## 4.2 Real-Time Line Movement Tracker

### Why

Improvement #8 ("Real-Time Line Movement") observes that lines that move 2+ points from opening to tip-off indicate sharp money or injury news. A line moving *toward* the model's prediction confirms the model; a line moving *away* from the model suggests new information the model doesn't have. The dossier's market cross-check (§11) used this principle to identify 3 games where market sentiment diverged from the statistical model.

### Expected Effect

- **Value betting**: Tracking opening vs. current lines reveals when the market agrees or disagrees with the model. Sustained disagreement (model says -7, line moves from -5 to -3) is a signal to defer to the market.
- **Injury inference**: Sharp line movement without public news often means injury information is circulating in betting circles. The tracker can flag these automatically.
- **UI**: A new "Line Movement" column in the All Games tab shows the direction and magnitude of movement.

### Implementation

Extend `opening_line_database.py`:

```python
from datetime import datetime
from pathlib import Path
import json

LINE_HISTORY_PATH = Path('data_files/line_movement_history.json')


def record_line_snapshot(game_id: str, current_spread: float,
                         current_total: float, source: str = 'odds_api'):
    """Append a timestamped line snapshot for a game.

    Stored as JSON: {game_id: [{timestamp, spread, total, source}, ...]}
    """
    history = {}
    if LINE_HISTORY_PATH.exists():
        with open(LINE_HISTORY_PATH) as f:
            history = json.load(f)

    if game_id not in history:
        history[game_id] = []

    history[game_id].append({
        'timestamp': datetime.utcnow().isoformat() + 'Z',
        'spread': current_spread,
        'total': current_total,
        'source': source,
    })

    with open(LINE_HISTORY_PATH, 'w') as f:
        json.dump(history, f, indent=2)


def get_line_movement(game_id: str) -> dict:
    """Calculate line movement from opening to latest snapshot.

    Returns:
        Dict with opening_spread, current_spread, spread_movement,
        opening_total, current_total, total_movement, snapshots_count.
    """
    if not LINE_HISTORY_PATH.exists():
        return {}

    with open(LINE_HISTORY_PATH) as f:
        history = json.load(f)

    snapshots = history.get(game_id, [])
    if len(snapshots) < 2:
        return {}

    opening = snapshots[0]
    latest = snapshots[-1]

    return {
        'opening_spread': opening['spread'],
        'current_spread': latest['spread'],
        'spread_movement': latest['spread'] - opening['spread'],
        'opening_total': opening['total'],
        'current_total': latest['total'],
        'total_movement': latest['total'] - opening['total'],
        'snapshots_count': len(snapshots),
        'first_seen': opening['timestamp'],
        'last_seen': latest['timestamp'],
    }


def detect_sharp_movement(game_id: str, threshold: float = 2.0) -> dict:
    """Flag games with significant line movement.

    Args:
        game_id: Game identifier.
        threshold: Minimum point movement to flag (default 2.0).

    Returns:
        Dict with is_sharp, direction, magnitude, and alert message.
    """
    movement = get_line_movement(game_id)
    if not movement:
        return {'is_sharp': False}

    spread_move = abs(movement.get('spread_movement', 0))
    total_move = abs(movement.get('total_movement', 0))

    is_sharp = spread_move >= threshold or total_move >= threshold * 1.5

    direction = ''
    if movement.get('spread_movement', 0) > 0:
        direction = 'toward underdog'
    elif movement.get('spread_movement', 0) < 0:
        direction = 'toward favorite'

    return {
        'is_sharp': is_sharp,
        'spread_direction': direction,
        'spread_magnitude': movement.get('spread_movement', 0),
        'total_magnitude': movement.get('total_movement', 0),
        'alert': f"⚡ Sharp move: spread {movement.get('spread_movement', 0):+.1f} ({direction})"
            if is_sharp else '',
    }
```

Schedule line snapshots in GitHub Actions:

```yaml
  # Add to update-efficiency-ratings.yml or a new workflow
  - name: Capture current odds
    run: python -c "
      from fetch_live_odds import fetch_current_odds
      from opening_line_database import record_line_snapshot
      odds = fetch_current_odds()
      for game in odds:
          record_line_snapshot(game['game_id'], game.get('spread', 0), game.get('total', 0))
      print(f'Recorded {len(odds)} line snapshots')
    "
    continue-on-error: true
```

---

## 4.3 Injury Monitoring Integration

### Why

Improvement #3 ("Injury / Suspension Monitor") identifies that a single player's absence can shift a team's efficiency by 2–5 points. The dossier cited the example of a hypothetical Kentucky star missing the tournament — a 4-point swing that would shift Kentucky from a favorite to a slight underdog in many matchups. The current model has no mechanism to account for player availability.

### Expected Effect

- **Spread model**: Manual or scraped injury data can shift predicted spreads by 1–5 points, which is often larger than the model's margin of error.
- **Value betting**: If the model knows about an injury before the market fully prices it in, the resulting edge can be 5–10 %.
- **Bracket simulation**: Injured star absence should reduce a team's composite rating, cascading through all rounds.

### Implementation

Create an injury adjustment system:

```python
# injury_monitor.py

import json
from pathlib import Path
from datetime import datetime

INJURY_FILE = Path('data_files/injuries.json')

# Impact estimates by player role (points of efficiency shift)
ROLE_IMPACT = {
    'star':    4.0,   # Leading scorer / primary ball handler
    'starter': 2.0,   # Regular starter
    'rotation': 0.8,  # Key rotation player
    'bench':   0.2,   # Deep bench
}


def load_injuries() -> dict:
    """Load current injury report.

    Format: {team_name: [{player, role, status, impact_override, updated}]}
    """
    if INJURY_FILE.exists():
        with open(INJURY_FILE) as f:
            return json.load(f)
    return {}


def get_team_injury_adjustment(team_name: str) -> float:
    """Calculate net efficiency adjustment for a team's injuries.

    Returns:
        Negative value (efficiency penalty) for injured players.
        E.g., -3.5 means the team is estimated 3.5 points worse per 100 possessions.
    """
    injuries = load_injuries()
    team_injuries = injuries.get(team_name, [])

    total_adjustment = 0.0
    for injury in team_injuries:
        status = injury.get('status', '').lower()
        if status in ('out', 'doubtful'):
            impact = injury.get('impact_override',
                                ROLE_IMPACT.get(injury.get('role', 'bench'), 0.2))
            total_adjustment -= impact
        elif status == 'questionable':
            impact = injury.get('impact_override',
                                ROLE_IMPACT.get(injury.get('role', 'bench'), 0.2))
            total_adjustment -= impact * 0.5  # 50% chance they play

    return total_adjustment


def apply_injury_adjustments(efficiency_data: dict) -> dict:
    """Apply injury adjustments to all teams' composite ratings.

    Modifies efficiency_data in place.

    Returns:
        Dict of teams that were adjusted and by how much.
    """
    adjustments = {}
    for team_name in efficiency_data:
        adj = get_team_injury_adjustment(team_name)
        if adj != 0.0:
            if 'composite_net' in efficiency_data[team_name]:
                efficiency_data[team_name]['composite_net'] += adj
            if 'adj_offense' in efficiency_data[team_name]:
                # Split adjustment 60/40 offense/defense
                efficiency_data[team_name]['adj_offense'] += adj * 0.6
                efficiency_data[team_name]['adj_defense'] -= adj * 0.4
            adjustments[team_name] = adj

    return adjustments
```

Create `data_files/injuries.json` (manually updated or scraped):

```json
{
    "Kentucky": [
        {
            "player": "Example Player",
            "role": "star",
            "status": "out",
            "impact_override": null,
            "updated": "2026-03-18"
        }
    ]
}
```

---

## 4.4 Selection Sunday Topology Update

### Why

Improvement #1 ("Selection Sunday Topology Update") addresses the fact that bracket structure (which teams are in which region, and the specific first-round matchups) is only known on Selection Sunday, roughly 4 days before the tournament starts. The system needs to ingest the actual bracket topology as soon as it's announced and re-run all simulations with the real structure.

### Expected Effect

- **Bracket simulation**: Real bracket topology replaces placeholder/projected brackets, immediately making all simulation output actionable.
- **Matchup analysis**: Once the real bracket is known, all Phase 2 matchup features can be computed for actual pairs.

### Implementation

Create a bracket loader:

```python
# bracket_loader.py

import json
from pathlib import Path
from bracket_simulation import Team, BracketState

BRACKET_FILE = Path('data_files/tournament_bracket.json')


def load_bracket_from_file(path: str = None) -> BracketState:
    """Load the official bracket from a JSON file.

    Expected format:
    {
        "year": 2026,
        "regions": {
            "East": [
                {"seed": 1, "team": "Duke", "id": "duke"},
                {"seed": 16, "team": "Norfolk State", "id": "norfolk-state"},
                ...
            ],
            ...
        }
    }
    """
    bracket_path = Path(path) if path else BRACKET_FILE
    with open(bracket_path) as f:
        data = json.load(f)

    teams = {}
    regions = {'East': [], 'West': [], 'Midwest': [], 'South': []}

    for region_name, team_list in data.get('regions', {}).items():
        for t in team_list:
            team = Team(
                id=t['id'],
                name=t['team'],
                seed=t['seed'],
                region=region_name,
                stats=t.get('stats', {}),
            )
            teams[team.id] = team
            if region_name in regions:
                regions[region_name].append(team)

    return BracketState(teams=teams, regions=regions)


def enrich_bracket_with_ratings(
    bracket: BracketState,
    efficiency_data: dict,
    composite_data: dict = None,
) -> BracketState:
    """Attach efficiency ratings and composite scores to bracket teams.

    Args:
        bracket: BracketState with teams loaded.
        efficiency_data: Dict mapping team names to efficiency dicts.
        composite_data: Dict mapping team names to composite rating dicts.

    Returns:
        Same BracketState with stats populated.
    """
    for team in bracket.teams.values():
        eff = efficiency_data.get(team.name, {})
        team.stats.update(eff)

        if composite_data:
            comp = composite_data.get(team.name, {})
            team.stats.update(comp)

        # Compute composite_rating for logistic predictor
        team.stats['composite_rating'] = team.stats.get('composite_net', 0)

    return bracket
```

---

## 4.5 Conference Tournament Results Integration

### Why

Improvement #2 ("Conference Tournament Results") notes that conference tournament performance in the 1–2 weeks before Selection Sunday provides the most recent signal about a team's form. A team that wins 4 games in 4 days to earn an auto-bid is both proven under pressure and potentially fatigued (feeding into Phase 2's fatigue feature).

### Implementation

```python
# In feature_engineering.py or a new conf_tourney.py

def get_conf_tourney_results(team_name: str, schedule_data: list) -> dict:
    """Extract conference tournament performance for a team.

    Args:
        team_name: Team to look up.
        schedule_data: List of game dicts with keys: date, opponent, result,
                       is_conf_tourney.

    Returns:
        Dict with conf_tourney_games, conf_tourney_wins, conf_tourney_ppg,
        conf_tourney_margin, is_auto_bid.
    """
    conf_games = [g for g in schedule_data
                  if g.get('is_conf_tourney') and g.get('team') == team_name]

    if not conf_games:
        return {
            'conf_tourney_games': 0,
            'conf_tourney_wins': 0,
            'conf_tourney_ppg': 0.0,
            'conf_tourney_margin': 0.0,
            'is_auto_bid': False,
        }

    wins = sum(1 for g in conf_games if g.get('result') == 'W')
    points = [g.get('points_for', 0) for g in conf_games]
    margins = [g.get('points_for', 0) - g.get('points_against', 0) for g in conf_games]

    return {
        'conf_tourney_games': len(conf_games),
        'conf_tourney_wins': wins,
        'conf_tourney_ppg': sum(points) / len(points) if points else 0.0,
        'conf_tourney_margin': sum(margins) / len(margins) if margins else 0.0,
        'is_auto_bid': wins == len(conf_games),  # Won every game = likely auto-bid
    }
```

---

## Deployment & Scheduling

| Component | Trigger | Frequency |
|-----------|---------|-----------|
| Efficiency rating update | GitHub Actions cron | 2× daily during tournament |
| Line snapshot capture | GitHub Actions cron | Every 4 hours during tournament |
| Injury report update | Manual edit of `injuries.json` | As needed |
| Bracket topology load | Manual after Selection Sunday | Once per year |
| Composite rating rebuild | After any rating update | Automatic |

---

## Files Modified / Created

| File | Change |
|------|--------|
| `.github/workflows/update-efficiency-ratings.yml` | Enhanced schedule, composite build step, metadata recording |
| `opening_line_database.py` | Add `record_line_snapshot`, `get_line_movement`, `detect_sharp_movement` |
| `injury_monitor.py` | **New** — injury tracking and efficiency adjustment |
| `bracket_loader.py` | **New** — official bracket ingestion |
| `predictions.py` | Data freshness indicator in sidebar |
| `data_files/injuries.json` | **New** — current injury report |
| `data_files/tournament_bracket.json` | **New** — official bracket data |
| `data_files/line_movement_history.json` | **New** — historical line snapshots |
| `data_files/ratings_metadata.json` | **New** — update timestamps |
