# march-madness
March Madness College Basketball Predictions

## Overview
A comprehensive March Madness betting prediction system that combines historical data, machine learning models, and real-time game schedules to provide accurate betting predictions for tournament games. Now enhanced with KenPom and BartTorvik efficiency ratings for superior prediction accuracy.

## Features
- **Comprehensive Data Collection**: 10 years of historical tournament data (2016-2025)
- **Advanced Efficiency Metrics**: KenPom and BartTorvik ratings integration (364 teams)
- **Enhanced ML Models**: 11-feature models with 8.4% spread accuracy improvement
- **Real-time Predictions**: Live game schedules with betting predictions
- **Underdog Value Bets**: Automatic detection of profitable underdog opportunities
- **Kelly Criterion Betting**: Optimal bet sizing recommendations
- **Multi-API Integration**: CBBD + ESPN + KenPom + BartTorvik
- **Streamlit UI**: Interactive web interface for predictions

## Recent Updates (v2.0)

### 🎯 Model Performance Improvements
- **Spread Accuracy**: 8.4% improvement (-0.97 points MAE)
- **Moneyline Accuracy**: +2.7 percentage points
- **Total Accuracy**: 1.7% improvement
- **Feature Expansion**: 3 → 11 features (CBBD + KenPom + BartTorvik)

### 🔧 New Data Sources
- **KenPom**: Advanced efficiency ratings (NetRtg, ORtg, DRtg, AdjT, Luck, SOS)
- **BartTorvik**: Adjusted offensive/defensive efficiency
- **Team Canonicalization**: 99.7% mapping coverage (364/365 teams)
- **Automated Data Pipeline**: Selenium scraping + data enrichment

### 📊 Enhanced Features
- **Extended Feature Set**: 6 KenPom + 2 BartTorvik metrics
- **Canonical Team Names**: Standardized naming across all sources
- **Data Validation**: Complete feature coverage for 15,961 games
- **Model Retraining**: Optimized for extended feature space

## API Information
This project uses the [College Basketball Data API](https://api.collegebasketballdata.com/) to fetch data. The API key provides 1,000 calls per month.

## Setup

### Prerequisites
- Python 3.8+
- CBBD API key (set as `CBBD_API_KEY` environment variable)

### Installation
```bash
pip install -r requirements.txt
```

### Environment Setup
```bash
export CBBD_API_KEY="your_api_key_here"
```

## Data Collection

### Comprehensive Historical Data
Collect 10 years of betting data for model training:

```bash
python data_collection.py --collect
```

This collects:
- Tournament games (2016-2025)
- Betting lines and odds
- Team season statistics
- Efficiency ratings
- Poll rankings

### KenPom & BartTorvik Data
Automated collection of advanced efficiency metrics:

```bash
# Download KenPom ratings
python download_kenpom.py

# Download BartTorvik ratings (auto-renames and cleans duplicates)
python download_barttorvik.py

# Create canonical datasets
python data_tools/efficiency_loader.py
```

**Output files:**
- `data_files/kenpom_ratings.csv` - Raw KenPom data (365 teams)
- `data_files/barttorvik_ratings.csv` - Raw BartTorvik data (365 teams, auto-renamed from download)
- `data_files/kenpom_canonical.csv` - Cleaned with canonical names (364 teams)
- `data_files/barttorvik_canonical.csv` - Cleaned with canonical names (364 teams)

### Team Name Canonicalization
Automatic mapping of team names across data sources:

```bash
# Match team names and create mappings
python scripts/match_teams.py

# Apply manual corrections
python scripts/apply_mappings.py
```

**Output files:**
- `data_files/kenpom_to_espn_matches.csv` - KenPom → canonical mappings
- `data_files/bart_to_espn_matches.csv` - BartTorvik → canonical mappings

### Data Enrichment
Enhance training data with advanced metrics:

```bash
# Add KenPom/BartTorvik features to historical games
python enrich_training_data.py

# Retrain models with extended features
python retrain_with_extended_features.py
```

**Output files:**
- `data_files/training_data_enriched.csv` - Historical data with all features
- `data_files/training_data_complete_features.csv` - Complete cases only (15,961 games)
- `data_files/models/` - Retrained models with 11 features

### Individual Data Collection Functions

#### Tournament Games
```python
from data_collection import fetch_all_tournament_games

# Get games for specific years
games = fetch_all_tournament_games(2023, 2024)
print(f"Found {len(games)} tournament games")
```

#### Betting Lines
```python
from data_collection import fetch_historical_lines

# Get betting lines for tournaments
lines = fetch_historical_lines(2023, 2024)
print(f"Found {len(lines)} betting lines")
```

#### Rankings
```python
from data_collection import fetch_rankings

# Get poll rankings
rankings = fetch_rankings(2024)
print(f"Found {len(rankings)} ranking entries")
```

#### Team Statistics
```python
from data_collection import fetch_team_season_data, fetch_efficiency_ratings

# Get team stats and efficiency ratings
team_stats = fetch_team_season_data(2024)
efficiency = fetch_efficiency_ratings(2024)
```

### Examples
See `examples/data_collection_examples.py` for complete usage examples.

## Automated Data Collection

### Automated Efficiency Data Updates
The system automatically fetches KenPom and BartTorvik efficiency ratings daily at 2 AM ET, but only during basketball season and only when games are scheduled.

**When it runs:**
- Basketball season: October 15 - April 15
- Games scheduled: Today or tomorrow have college basketball games
- Time: Daily at 2 AM Eastern Time (with random 0-60 minute delay)

**Anti-detection measures:**
- Randomized timing to avoid consistent patterns
- Headless Chrome with anti-bot countermeasures
- Custom user agents and browser fingerprinting evasion
- Random delays between site requests
- Only runs when actually needed (season + games scheduled)

**What gets updated:**
- `data_files/kenpom_ratings.csv` - Raw KenPom data
- `data_files/barttorvik_ratings.csv` - Raw BartTorvik data
- `data_files/kenpom_canonical.csv` - Cleaned KenPom with canonical names
- `data_files/barttorvik_canonical.csv` - Cleaned BartTorvik with canonical names

**Manual trigger:**
You can also manually run the efficiency data update:
```bash
python check_efficiency_update_needed.py && python download_kenpom.py && python download_barttorvik.py && python data_tools/efficiency_loader.py
```

**GitHub Actions Workflow:**
The automated workflow (`.github/workflows/update-efficiency-ratings.yml`):
- Runs daily at 2 AM ET (7 AM UTC) with randomization
- Only executes during basketball season with scheduled games
- Includes anti-detection measures to avoid blocking
- Commits changes back to the repository
- Only commits if efficiency data has actually changed

## Underdog Value Bets

### Finding Profitable Opportunities
The system automatically identifies underdog betting opportunities where the model predicts a higher win probability than the betting odds suggest.

```python
from underdog_value import identify_underdog_value

# Example: Model gives underdog 40% chance, but odds imply 27%
game = {
    'home_team': 'Kansas',
    'away_team': 'NC State',
    'home_moneyline': -350,
    'away_moneyline': +275
}

home_win_prob = 0.60  # Model prediction
value_bet = identify_underdog_value(game, home_win_prob, min_ev_threshold=5.0)

if value_bet:
    print(f"Value bet: {value_bet['team']}")
    print(f"Edge: {value_bet['edge']:.1%}")
    print(f"ROI: {value_bet['roi']:.1f}%")
```

### Kelly Criterion Bet Sizing
```python
from underdog_value import get_betting_recommendation

recommendation = get_betting_recommendation(value_bet, bankroll=1000)
print(f"Recommended bet: ${recommendation['recommended_bet']:.2f}")
print(f"Kelly %: {recommendation['kelly_percentage']:.1f}%")
```

### Examples
See `examples/underdog_value_examples.py` for detailed examples.

## Running Predictions

### Streamlit UI
```bash
streamlit run predictions.py
```

### ESPN Data Integration
The system automatically fetches current season games from ESPN and enriches them with CBBD statistics for accurate predictions.

## Machine Learning Models

### Extended Feature Set (v2.0)
Models trained with 11 features (vs. original 3):

**CBBD Features (3):**
- `team_season_win_pct` - Team's season win percentage
- `team_season_avg_mov` - Average margin of victory
- `opponent_season_avg_mov` - Opponent's average margin of victory

**KenPom Features (6):**
- `net_rtg` - Net rating (offensive - defensive)
- `off_rtg` - Offensive rating
- `def_rtg` - Defensive rating
- `adj_tempo` - Adjusted tempo
- `luck` - Luck rating
- `sos` - Strength of schedule

**BartTorvik Features (2):**
- `adj_oe` - Adjusted offensive efficiency
- `adj_de` - Adjusted defensive efficiency

### Supported Models
- **XGBoost**: Gradient boosting for complex patterns
- **Random Forest**: Ensemble learning for robust predictions
- **Linear Regression**: Spread and total predictions
- **Logistic Regression**: Moneyline predictions

### Performance Improvements (v2.0)
Significant accuracy gains with extended features:

| Metric | Baseline (3 features) | Extended (11 features) | Improvement |
|--------|----------------------|----------------------|-------------|
| Spread MAE | 12.8 | 11.7 | 8.4% better |
| Moneyline Accuracy | 68.2% | 70.9% | +2.7 pts |
| Total MAE | 14.2 | 13.9 | 1.7% better |

**Training Data:** 15,961 complete games (2016-2024 seasons)

### Training Process
```bash
# Retrain with extended features
python retrain_with_extended_features.py
```

**Training Details:**
- Cross-validation with 5 folds
- Ensemble of XGBoost, Random Forest, Linear/Logistic Regression
- Separate models for spread, total, and moneyline predictions
- Feature scaling and preprocessing

## Data Sources

### College Basketball Data API (CBBD)
- Historical tournament games (2016-2025)
- Betting lines and odds
- Team season statistics
- Efficiency ratings
- Poll rankings

### ESPN API
- Current season game schedules
- Real-time game information
- Team name standardization

## Project Structure
```
march-madness/
├── predictions.py                    # Streamlit UI and prediction logic
├── data_collection.py               # CBBD API integration and data collection
├── fetch_espn_cbb_scores.py         # ESPN API scraper
├── model_training.py                # Original model training (3 features)
├── retrain_with_extended_features.py # Extended model training (11 features)
├── enrich_training_data.py          # Add KenPom/BartTorvik features
├── download_kenpom.py               # KenPom data scraper
├── download_barttorvik.py           # BartTorvik data scraper
├── underdog_value.py                # Value bet detection and Kelly sizing
├── generate_predictions.py          # Generate predictions for upcoming games
├── display_predictions.py           # Display prediction results
├── data_tools/
│   ├── efficiency_loader.py         # Load and clean efficiency data
│   └── __init__.py
├── scripts/
│   ├── match_teams.py               # Team name canonicalization
│   ├── apply_mappings.py            # Apply manual team mappings
│   └── __init__.py
├── examples/                        # Usage examples
│   ├── data_collection_examples.py
│   └── underdog_value_examples.py
├── data_files/                      # Cached data and models
│   ├── cache/                      # API response cache
│   ├── models/                     # Trained ML models
│   ├── espn_cbb_current_season.csv # Current season games
│   ├── training_data_complete_features.csv # Training data (15,961 games)
│   ├── kenpom_canonical.csv        # Cleaned KenPom data (364 teams)
│   ├── barttorvik_canonical.csv    # Cleaned BartTorvik data (364 teams)
│   └── upcoming_game_predictions.json # AI predictions
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── copilot-instructions.md          # AI assistant guidelines
```

## Roadmap Implementation

### ✅ Completed Features (v2.0)
- [x] Comprehensive data collection functions
- [x] Historical betting data (10 years)
- [x] ML model training pipeline
- [x] ESPN integration for current games
- [x] Streamlit prediction interface
- [x] Team name normalization
- [x] Data caching and compression
- [x] **Underdog value bet detection**
- [x] **Kelly Criterion bet sizing**
- [x] **Expected value calculations**
- [x] **KenPom efficiency ratings integration** (6 features)
- [x] **BartTorvik advanced metrics integration** (2 features)
- [x] **Extended ML models with 11 features** (8.4% spread improvement)
- [x] **Automated team name canonicalization** (99.7% coverage)
- [x] **Enhanced data enrichment pipeline**
- [x] **Cross-validation training with performance metrics**

### 🔄 Next Steps
- [ ] Advanced betting features (road/neutral advantages)
- [ ] Model evaluation and comparison
- [ ] Real-time odds integration
- [ ] Prediction confidence scoring
- [ ] Live game tracking and in-game predictions

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License
See LICENSE file for details.
