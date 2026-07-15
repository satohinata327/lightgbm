# MF2レジーム相関ベースラインLightGBM

特徴量は`detection_light-gbm.ipynb`のベースライン特徴量セットと同一である。

## 評価指標

- 学習：AUC=1.0000、正解率=0.9940、Logloss=0.0806、n=664
- Validation：AUC=0.9788、正解率=0.9050、Logloss=0.2277、n=221
- Test：AUC=0.7516、正解率=0.8874、Logloss=0.2907、n=222

## 重要度上位の特徴量

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
