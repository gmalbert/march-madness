# Model Retraining Results - Extended Features

## Summary

Successfully retrained all prediction models with extended features including KenPom and BartTorvik efficiency ratings. Models now use **11 features** instead of 3.

## Performance Improvement

### 🎯 SPREAD PREDICTION
- **Baseline MAE** (3 features): **11.61 points**
- **Extended MAE** (11 features): **10.64 points**
- **Improvement**: **0.97 points (-8.4%)**
- **Impact**: Predictions are now **~1 point more accurate** on average

### 🎯 TOTAL PREDICTION
- **Baseline MAE** (3 features): **16.00 points**
- **Extended MAE** (11 features): **15.72 points**
- **Improvement**: **0.28 points (-1.7%)**
- **Impact**: Modest improvement, totals are inherently harder to predict

### 🎯 MONEYLINE PREDICTION
- **Baseline Accuracy** (3 features): **65.3%**
- **Extended Accuracy** (11 features): **68.0%**
- **Improvement**: **+2.7 percentage points**
- **Impact**: Correctly predicts **3 more winners out of every 100 games**

## Feature Set

### Baseline Features (3)
1. `off_eff_diff` - Offensive efficiency differential (CBBD)
2. `def_eff_diff` - Defensive efficiency differential (CBBD)
3. `net_eff_diff` - Net efficiency differential (CBBD)

### Extended Features Added (8)

#### KenPom Features (6)
4. `kenpom_netrtg_diff` - KenPom net rating differential
5. `kenpom_ortg_diff` - KenPom offensive rating differential
6. `kenpom_drtg_diff` - KenPom defensive rating differential
7. `kenpom_adjt_diff` - Adjusted tempo differential
8. `kenpom_luck_diff` - Luck factor differential
9. `kenpom_sos_diff` - Strength of schedule differential

#### BartTorvik Features (2)
10. `bart_oe_diff` - BartTorvik adjusted offensive efficiency differential
11. `bart_de_diff` - BartTorvik adjusted defensive efficiency differential

## Training Data

- **Total Games**: 15,961 (with complete features)
- **Years**: 2016-2025
- **Regular Season Games**: 15,233
- **Tournament Games**: 728
- **Feature Coverage**: 62.2% of all historical games have KenPom data, 61.3% have BartTorvik data

## Model Performance by Type

### Spread Prediction (Individual Models)
| Model | Baseline MAE | Extended MAE | Improvement |
|-------|--------------|--------------|-------------|
| Linear Regression | 11.38 | 10.20 | -1.18 (-10.4%) |
| Random Forest | 12.11 | 10.86 | -1.25 (-10.3%) |
| XGBoost | 11.35 | 10.86 | -0.49 (-4.3%) |

### Total Prediction (Individual Models)
| Model | Baseline MAE | Extended MAE | Improvement |
|-------|--------------|--------------|-------------|
| Linear Regression | 15.24 | 15.23 | -0.01 (-0.1%) |
| Random Forest | 17.04 | 15.97 | -1.07 (-6.3%) |
| XGBoost | 15.73 | 15.98 | +0.25 (+1.6%) |

### Moneyline Prediction (Individual Models)
| Model | Baseline Acc | Extended Acc | Improvement |
|-------|--------------|--------------|-------------|
| Logistic Regression | 67.9% | 69.5% | +1.6 pp |
| Random Forest | 62.3% | 67.2% | +4.9 pp |
| XGBoost | 65.6% | 67.2% | +1.6 pp |

## Key Insights

1. **Spread Predictions Show Biggest Improvement**: 8.4% reduction in MAE is substantial. The KenPom tempo and strength of schedule features appear particularly valuable for spread prediction.

2. **Random Forest Benefits Most**: Random Forest models showed the largest improvements across all three prediction types, suggesting it's particularly good at exploiting the additional features.

3. **Linear Models Remain Strong**: Despite having fewer parameters, linear regression with proper feature scaling performs competitively, especially for spread prediction.

4. **Moneyline Accuracy Boost**: The 2.7 percentage point improvement in win/loss prediction accuracy is significant - this translates to ~3 more correct predictions per 100 games.

5. **Total Prediction Less Sensitive**: Total points are inherently harder to predict and show less sensitivity to advanced metrics. This makes sense as totals depend more on game flow and situational factors.

## Production Status

✅ **Models Updated**: Extended feature models are now saved in `data_files/models/`
✅ **Predictions.py Updated**: Now uses 11 features for all predictions
✅ **Backward Compatible**: Falls back to zeros for missing KenPom/BartTorvik data
✅ **Ready for Production**: All models retrained and tested

## Next Steps

1. ✅ Models retrained with extended features
2. ✅ Predictions.py updated to use 11 features
3. ⏳ Monitor performance on upcoming games
4. ⏳ Potential feature engineering (interaction terms, ratios)
5. ⏳ Experiment with ensemble weighting based on feature availability

## Files Modified

- `retrain_with_extended_features.py` - Training script with comparison
- `enrich_training_data.py` - Adds KenPom/BartTorvik to historical data
- `predictions.py` - Updated to use 11-feature models
- `data_files/models/` - New model files with extended features
- `data_files/training_data_enriched.csv` - Historical data with all features
- `data_files/models/model_comparison.json` - Detailed comparison metrics

---

**Impact Summary**: The integration of KenPom and BartTorvik data has measurably improved model performance, particularly for spread and moneyline predictions. The system now leverages multiple expert rating systems to make more informed predictions.
