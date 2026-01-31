# Pre-Computed Bracket Simulations

## Overview

To improve performance and avoid timeouts on Streamlit Cloud, tournament bracket simulations are pre-computed and stored as JSON files. This eliminates the 10-30 second delay from running Monte Carlo simulations on every page load.

## How It Works

### 1. Pre-Computation (GitHub Actions)
- **Workflow**: `.github/workflows/precompute_brackets.yml`
- **Script**: `scripts/precompute_bracket_simulations.py`
- **Schedule**: Runs daily at 6 AM UTC
- **Output**: JSON files in `data_files/precomputed_brackets/`

The workflow:
1. Loads bracket data and efficiency ratings (KenPom/BartTorvik)
2. Runs 10,000 Monte Carlo simulations per tournament year
3. Saves results to `bracket_{year}.json`
4. Commits files back to the repository

### 2. Loading Results (Streamlit App)
- **File**: `pages/01_🏀_Tournament_Bracket.py`
- **Function**: `load_precomputed_bracket(year)`

The app:
1. First tries to load pre-computed results from JSON
2. Falls back to live simulation if file doesn't exist
3. Shows indicator of data source (pre-computed vs live)

## Performance Impact

| Method | Load Time | Simulations |
|--------|-----------|-------------|
| **Pre-computed** | ~0.5s | 10,000 |
| Live simulation | 10-30s | 1,000 |

Pre-computed results are **20-60x faster** and use **10x more simulations** for better accuracy.

## Manual Pre-Computation

Run locally or in CI:

```bash
# Default: 10,000 simulations for years 2025, 2024, 2023
python scripts/precompute_bracket_simulations.py

# Custom simulation count
python scripts/precompute_bracket_simulations.py --simulations 50000

# Specific years
python scripts/precompute_bracket_simulations.py --years 2025 2024
```

## Trigger GitHub Actions Workflow

1. Go to Actions tab in GitHub
2. Select "Pre-compute Tournament Brackets"
3. Click "Run workflow"
4. Optionally set simulation count (default: 10,000)

## File Format

```json
{
  "year": 2025,
  "num_simulations": 10000,
  "computed_at": "2026-01-31T12:00:00",
  "bracket_data": { ... },
  "simulation_results": {
    "team_1": {
      "team": {
        "name": "Duke",
        "seed": 1,
        "region": "East"
      },
      "round_32_prob": 0.985,
      "sweet_16_prob": 0.892,
      "elite_8_prob": 0.745,
      "final_four_prob": 0.523,
      "championship_prob": 0.289,
      "winner_prob": 0.156
    },
    ...
  }
}
```

## Updating Simulations

Simulations are automatically refreshed:
- **Daily** via GitHub Actions
- **On demand** via manual workflow trigger

To force immediate refresh:
1. Delete old JSON files
2. Run workflow manually
3. Wait for commit (usually 2-5 minutes)
4. Streamlit will use new data on next deployment

## Fallback Behavior

If pre-computed data is unavailable:
1. App shows "Running live simulation" message
2. Runs 1,000 simulations (reduced count for speed)
3. Still displays bracket, just slower

This ensures the app always works even if:
- Pre-computation fails
- Files are missing
- New tournament year is added
