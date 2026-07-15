# MF2 Regime-Correlation Baseline LightGBM

Features match `detection_light-gbm.ipynb` baseline feature set.

## Metrics

- train: AUC=1.0000, accuracy=0.9960, logloss=0.1369, n=1004
- val: AUC=0.9929, accuracy=0.7254, logloss=0.3955, n=335
- test: AUC=0.8021, accuracy=0.5910, logloss=0.6758, n=335

## Top Features

- corr_sp_rate: 76
- rate_acf1: 58
- joint_large_move_95: 51
- rate_skew: 42
- sp_acf1: 30
- rate_abs_acf1: 25
- rate_kurt: 23
- sp_abs_acf1: 15
- rate_abs_acf5: 12
- sp_std: 10
- sp_q90_abs: 9
- sp_q95_abs: 8
- rate_std: 8
- sp_acf5: 8
- sp_abs_acf5: 8
