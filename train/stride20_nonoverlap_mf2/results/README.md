# MF2レジーム相関ベースラインLightGBM

特徴量は`detection_light-gbm.ipynb`のベースライン特徴量セットと同一である。

## 評価指標

- 学習：AUC=1.0000、正解率=0.9949、Logloss=0.1225、n=980
- Validation：AUC=0.9868、正解率=0.7649、Logloss=0.3326、n=285
- Test：AUC=0.6164、正解率=0.7018、Logloss=0.5490、n=285

## 重要度上位の特徴量

- rate_acf1: 74
- corr_sp_rate: 65
- joint_large_move_95: 61
- rate_kurt: 37
- sp_acf1: 29
- rate_abs_acf1: 28
- rate_skew: 19
- sp_abs_acf1: 12
- sp_q90_abs: 10
- sp_q95_abs: 9
- rate_q90_abs: 7
- sp_abs_acf5: 7
- rate_abs_acf5: 7
- rate_tail_ratio: 7
- sp_kurt: 6
