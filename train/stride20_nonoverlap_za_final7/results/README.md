# ZA final7非重複分割LightGBM

特徴量は`detection_light-gbm.ipynb`のベースライン特徴量セットと同一である。

## 評価指標

- 学習：AUC=0.9999、正解率=0.9959、Logloss=0.3612、n=980
- Validation：AUC=0.8682、正解率=0.6807、Logloss=0.5721、n=285
- Test：AUC=0.8436、正解率=0.6877、Logloss=0.5770、n=285

## 重要度上位の特徴量

- sp_q90_abs: 40
- corr_sp_rate: 30
- sp_abs_acf5: 24
- joint_large_move_95: 16
- rate_acf1: 13
- sp_std: 8
- sp_kurt: 7
- sp_q95_abs: 7
- sp_acf1: 7
- sp_mean: 6
- sp_q99_abs: 6
- rate_q90_abs: 6
- rate_tail_ratio: 6
- rate_kurt: 5
- rate_skew: 4
