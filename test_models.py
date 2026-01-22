import pandas as pd
import joblib
import numpy as np

# Load the training data to get some sample games
df = pd.read_csv('data_files/training_data_weighted.csv')

# Load models
spread_model = joblib.load('data_files/models/spread_xgboost.joblib')
total_model = joblib.load('data_files/models/total_xgboost.joblib')
moneyline_model = joblib.load('data_files/models/moneyline_xgboost.joblib')

# Get some sample games with betting lines
sample_games = df[df['betting_spread'].notna()].head(5)

print('🎯 Sample Model Predictions vs Actual Results:')
print('=' * 60)

for idx, game in sample_games.iterrows():
    features = np.array([[
        game['home_efficiency_diff'],
        game['away_efficiency_diff'],
        game['neutral_site']
    ]])

    # Spread prediction
    spread_pred = spread_model.predict(features)[0]
    spread_actual = game.get('actual_spread', 'N/A')

    # Total prediction
    total_pred = total_model.predict(features)[0]
    total_actual = game.get('actual_total', 'N/A')

    # Moneyline prediction
    moneyline_prob = moneyline_model.predict_proba(features)[0][1]
    moneyline_pred = 'Home' if moneyline_prob > 0.5 else 'Away'
    moneyline_actual = 'Home' if game.get('home_score', 0) > game.get('away_score', 0) else 'Away'

    print(f'{game["home_team"]} vs {game["away_team"]} ({game["season"]})')
    print(f'  Spread: Pred={spread_pred:.1f}, Actual={spread_actual}, Betting={game["betting_spread"]}')
    print(f'  Total:  Pred={total_pred:.1f}, Actual={total_actual}, Betting={game.get("betting_over_under", "N/A")}')
    print(f'  Money:  Pred={moneyline_pred} ({moneyline_prob:.1%}), Actual={moneyline_actual}')
    print()