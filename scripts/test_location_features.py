#!/usr/bin/env python3
"""Test the upset prediction model with location features."""

from upset_prediction import train_upset_model
import joblib
from pathlib import Path

print('Training upset prediction model WITH location features...')
predictor, metrics = train_upset_model()

print('\nModel Performance:')
print(f'Train Accuracy: {metrics["train_accuracy"]:.4f}')
print(f'Test Accuracy: {metrics["test_accuracy"]:.4f}')
print(f'Precision: {metrics["precision"]:.4f}')
print(f'Recall: {metrics["recall"]:.4f}')
print(f'F1 Score: {metrics["f1_score"]:.4f}')

print('\nTop Feature Importances:')
importances = sorted(metrics['feature_importance'].items(), key=lambda x: x[1], reverse=True)
for feature, importance in importances:
    print(f'  {feature}: {importance:.4f}')

# Save the model
MODEL_DIR = Path('data_files/models')
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(predictor.model, MODEL_DIR / 'upset_predictor.joblib')
joblib.dump(predictor.scaler, MODEL_DIR / 'upset_predictor_scaler.joblib')
print(f'\nModel saved to {MODEL_DIR}')
