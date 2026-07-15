# MF2 Regime-Correlation Baseline LightGBM

Features match `detection_light-gbm.ipynb` baseline feature set.

## Metrics

- train: AUC=1.0000, accuracy=0.9940, logloss=0.0806, n=664
- val: AUC=0.9788, accuracy=0.9050, logloss=0.2277, n=221
- test: AUC=0.7516, accuracy=0.8874, logloss=0.2907, n=222

## Top Features

- joint_large_move_95: 105
- corr_sp_rate: 92
- rate_acf1: 80
- rate_skew: 52
- rate_abs_acf1: 25
- rate_kurt: 24
- sp_acf1: 21
- rate_abs_acf5: 21
- sp_acf5: 17
- rate_std: 13
- sp_abs_acf1: 12
- sp_mean: 8
- sp_std: 7
- rate_tail_ratio: 6
- rate_mean: 5
