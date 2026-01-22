# march-madness
March Madness College Basketball Predictions

## Overview
A comprehensive March Madness betting prediction system that combines historical data, machine learning models, and real-time game schedules to provide accurate betting predictions for tournament games.

## Features
- **Comprehensive Data Collection**: 10 years of historical tournament data (2016-2025)
- **Machine Learning Models**: XGBoost, Random Forest, Linear/Logistic Regression
- **Real-time Predictions**: Live game schedules with betting predictions
- **Underdog Value Bets**: Automatic detection of profitable underdog opportunities
- **Kelly Criterion Betting**: Optimal bet sizing recommendations
- **Dual API Integration**: CBBD for historical data + ESPN for current season games
- **Streamlit UI**: Interactive web interface for predictions

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

### Supported Models
- **XGBoost**: Gradient boosting for complex patterns
- **Random Forest**: Ensemble learning for robust predictions
- **Linear Regression**: Spread and total predictions
- **Logistic Regression**: Moneyline predictions

### Model Features
- Team efficiency ratings (offensive/defensive)
- Season statistics (FG%, 3PT%, rebounds, etc.)
- Historical performance metrics
- Betting market data

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
├── data_collection.py      # CBBD API integration and data collection
├── predictions.py          # Streamlit UI and prediction logic
├── fetch_espn_cbb_scores.py # ESPN API scraper
├── examples/               # Usage examples
│   └── data_collection_examples.py
├── data_files/             # Cached data and models
│   ├── cache/             # API response cache
│   └── espn_cbb_current_season.csv
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Roadmap Implementation

### ✅ Completed Features
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

### 🔄 Next Steps
- [ ] Advanced betting features (road/neutral advantages)
- [ ] Model evaluation and comparison
- [ ] Real-time odds integration
- [ ] Prediction confidence scoring

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License
See LICENSE file for details.
