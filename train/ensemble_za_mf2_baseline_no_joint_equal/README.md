# Baseline＋jointなし：等重みsoft voting

学習済みの次の2モデルについて、実データ確率を0.5ずつ加重平均する。

- `stride20_blocked_cv_za_mf2`
- `stride20_blocked_cv_za_mf2_no_joint`

```text
prob_real = 0.5 * prob_baseline + 0.5 * prob_no_joint
```

0.5以上を実データと判定する。新しいLightGBMの学習は行わず、構成モデルのモデル・scaler・特徴量設定をそのまま使用する。
