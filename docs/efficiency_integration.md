# KenPom & BartTorvik Integration Guide

## Overview

This document describes the integration of KenPom and BartTorvik efficiency ratings into the March Madness prediction system.

## Data Pipeline

### 1. Data Collection
- **KenPom**: [download_kenpom.py](../download_kenpom.py)
  - Automated Selenium scraper
  - Extracts 365 Division I teams
  - 21 columns: NetRtg, ORtg, DRtg, AdjT, Luck, SOS metrics, NCSOS, and ranks
  - Output: `data_files/kenpom_ratings.csv`

- **BartTorvik**: [download_barttorvik.py](../download_barttorvik.py)
  - Automated Selenium scraper
  - CSV download from T-Rank website
  - Output: `data_files/barttorvik_ratings.csv`

### 2. Team Name Canonicalization
- **Mapping Process**: [scripts/match_teams.py](../scripts/match_teams.py)
  - Uses `normalize_team_name()` function from predictions.py
  - Fuzzy matching with 0.75 cutoff threshold
  - Special case handling (Miami FL→Miami, Connecticut→UConn, etc.)
  - Manual review workflow for ambiguous matches
  
- **Mapping Files**:
  - `data_files/kenpom_to_espn_matches.csv` - 364/365 teams mapped
  - `data_files/bart_to_espn_matches.csv` - 364/365 teams mapped
  - Only "New Haven" remains unmapped (99.7% coverage)

### 3. Data Loading
- **Loader**: [data_tools/efficiency_loader.py](../data_tools/efficiency_loader.py)
  - `EfficiencyDataLoader` class
  - Applies canonical team name mappings
  - Cleans numeric columns (removes '+' signs, converts ranks to int)
  - Parses W-L records into wins/losses/win_pct
  - Outputs canonical datasets for modeling

## Integration into Predictions

### Features Added

#### KenPom Features (6 features)
1. **NetRtg Diff**: Home NetRtg - Away NetRtg (overall team quality)
2. **ORtg Diff**: Home ORtg - Away ORtg (offensive efficiency)
3. **DRtg Diff**: Home DRtg - Away DRtg (defensive efficiency, lower is better)
4. **AdjT Diff**: Home AdjT - Away AdjT (tempo differential)
5. **Luck Diff**: Home Luck - Away Luck (over/underperformance vs Pythagorean)
6. **SOS_NetRtg Diff**: Home SOS - Away SOS (strength of schedule)

#### BartTorvik Features (2 features)
1. **Adj OE Diff**: Home Adj OE - Away Adj OE (offensive efficiency)
2. **Adj DE Diff**: Home Adj DE - Away Adj DE (defensive efficiency)

### Code Flow

```python
# In predictions.py main():
1. Load models
2. Load KenPom/BartTorvik data via get_kenpom_barttorvik_data()
3. For each game:
   a. Get advanced metrics via enrich_with_advanced_metrics()
   b. Pass to make_predictions() with advanced_metrics parameter
   c. calculate_features() computes extended features
   d. Current models still use original 3 CBBD features
   e. Extended features available for future model retraining
```

### Function Reference

#### `get_kenpom_barttorvik_data()`
Returns: `(kenpom_df, bart_df)` - DataFrames with canonical team names

#### `enrich_with_advanced_metrics(home_team_name, away_team_name, kenpom_df, bart_df)`
Returns: 
```python
{
    'home': {'kenpom': {...}, 'barttorvik': {...}},
    'away': {'kenpom': {...}, 'barttorvik': {...}}
}
```

#### `calculate_features(home_team, away_team, home_eff, away_eff, advanced_metrics)`
Returns:
```python
{
    'spread': [3 base features],  # Used by current models
    'total': [3 base features],   # Used by current models
    'moneyline': [3 base features],  # Used by current models
    'extended': [3 base + 6 kenpom + 2 bart = 11 features],  # For future use
    'kenpom_available': bool,
    'barttorvik_available': bool
}
```

## Current vs Future State

### Current (v1.0)
- ✅ Data collection automated (Selenium)
- ✅ Team name canonicalization (99.7% coverage)
- ✅ Data loader integrated
- ✅ Advanced metrics calculated
- ⚠️ Models still use original 3 CBBD features only
- ⚠️ Extended features computed but not used in predictions

### Future (v2.0) - Model Retraining
When ready to retrain models with extended features:

1. **Update Training Data**:
   ```python
   # In model training script
   from data_tools.efficiency_loader import EfficiencyDataLoader
   
   loader = EfficiencyDataLoader()
   kenpom_df = loader.load_kenpom()
   bart_df = loader.load_barttorvik()
   
   # Merge with historical games
   # Add KenPom/BartTorvik features for each game
   ```

2. **Extend Feature Set**:
   - Current: 3 features (off_eff_diff, def_eff_diff, net_eff_diff)
   - Future: 11 features (3 base + 6 KenPom + 2 BartTorvik)

3. **Update Prediction Code**:
   ```python
   # Change in calculate_features():
   'spread': extended_features,  # Use all 11 features
   'total': extended_features,
   'moneyline': extended_features
   ```

4. **Feature Column Names**:
   ```python
   spread_feature_names = [
       'off_eff_diff', 'def_eff_diff', 'net_eff_diff',  # CBBD
       'kenpom_netrtg_diff', 'kenpom_ortg_diff', 'kenpom_drtg_diff',  # KenPom
       'kenpom_adjt_diff', 'kenpom_luck_diff', 'kenpom_sos_diff',
       'bart_oe_diff', 'bart_de_diff'  # BartTorvik
   ]
   ```

## File Structure

```
march-madness/
├── data_files/
│   ├── kenpom_ratings.csv              # Raw KenPom data (365 teams)
│   ├── barttorvik_ratings.csv          # Raw BartTorvik data (365 teams)
│   ├── kenpom_canonical.csv            # Cleaned with canonical names (364 teams)
│   ├── barttorvik_canonical.csv        # Cleaned with canonical names (364 teams)
│   ├── kenpom_to_espn_matches.csv      # KenPom→ESPN name mappings
│   └── bart_to_espn_matches.csv        # BartTorvik→ESPN name mappings
├── data_tools/
│   └── efficiency_loader.py            # Data loader class
├── scripts/
│   ├── match_teams.py                  # Team name matching
│   ├── export_missing_teams.py         # Export unmapped teams
│   └── apply_mappings.py               # Apply manual mappings
├── docs/
│   ├── kenpom_fields.md                # KenPom field documentation
│   └── efficiency_integration.md       # This file
├── download_kenpom.py                  # KenPom scraper
├── download_barttorvik.py              # BartTorvik scraper
└── predictions.py                      # Main app with integration
```

## Usage

### Running Data Collection
```bash
# Download KenPom data
python download_kenpom.py

# Download BartTorvik data
python download_barttorvik.py

# Create canonical datasets
python data_tools/efficiency_loader.py
```

### Using Efficiency Data in Code
```python
from data_tools.efficiency_loader import EfficiencyDataLoader

# Initialize loader
loader = EfficiencyDataLoader()

# Get data for specific team
michigan_data = loader.get_merged_efficiency('Michigan')
print(michigan_data['kenpom']['NetRtg'])  # 35.99
print(michigan_data['barttorvik']['Adj OE'])  # 128.15

# Or load full datasets
kenpom_df = loader.load_kenpom()
bart_df = loader.load_barttorvik()

# Filter for specific team
michigan_kp = kenpom_df[kenpom_df['canonical_team'] == 'Michigan']
```

## Maintenance

### Updating Data During Season
1. Run scrapers periodically (weekly recommended)
2. Canonical datasets auto-update via loader
3. No need to re-run team matching (mappings are stable)

### Adding New Teams
If new teams appear mid-season:
1. Run `scripts/match_teams.py`
2. Check for new unmapped teams
3. Add manual mappings if needed
4. Run `scripts/apply_mappings.py`

### Troubleshooting

**Error: "Can only use .str accessor with string values"**
- KenPom numeric columns already have '+' signs in data
- Fixed: `clean_numeric()` function handles both object and numeric types

**Error: "Warning: Dropping X unmapped teams"**
- Expected for "New Haven" (not in canonical dataset)
- Other teams: check spelling, add manual mapping

**Missing Features**
- If `kenpom_available: False`, team not found in KenPom data
- Check canonical team name vs KenPom team name
- May need additional mapping entry

## Performance Impact

- **Data Loading**: ~2 seconds on first load (364 teams × 2 sources)
- **Per-Game Enrichment**: <10ms (DataFrame lookup)
- **Memory**: ~5MB for both datasets combined
- **Cached**: DataFrames loaded once at app startup

## Next Steps

1. ✅ Data collection automated
2. ✅ Team canonicalization complete
3. ✅ Integration into prediction pipeline
4. ⏳ **Model retraining with extended features** (future)
5. ⏳ A/B testing: CBBD-only vs CBBD+KenPom+BartTorvik
6. ⏳ Feature importance analysis
7. ⏳ Cross-validation with different feature combinations

## References

- [KenPom Fields Documentation](kenpom_fields.md)
- [KenPom Website](https://kenpom.com)
- [BartTorvik (T-Rank)](https://barttorvik.com/trank.php)
- [CBBD API Documentation](https://github.com/mcboog/cbbd)
