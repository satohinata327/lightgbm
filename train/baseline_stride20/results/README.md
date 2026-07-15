# MF2レジーム相関ベースラインLightGBM

特徴量は`detection_light-gbm.ipynb`のベースライン特徴量セットと同一である。

## 評価指標

- 学習：AUC=1.0000、正解率=0.9960、Logloss=0.1369、n=1004
- Validation：AUC=0.9929、正解率=0.7254、Logloss=0.3955、n=335
- Test：AUC=0.8021、正解率=0.5910、Logloss=0.6758、n=335

## 重要度上位の特徴量

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
