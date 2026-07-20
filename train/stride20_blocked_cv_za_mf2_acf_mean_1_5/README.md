# ZA＋MF2 baseline：lag 1〜5平均自己相関モデル

26特徴量baselineの自己相関8特徴量を、lag 1〜5の算術平均4特徴量へ置換した22特徴量モデルである。

```text
acf_mean_1_5(x)
= (acf(x, 1) + acf(x, 2) + acf(x, 3) + acf(x, 4) + acf(x, 5)) / 5
```

使用する平均自己相関は次の4個である。

- `sp_acf_mean_1_5`：SP500リターン
- `sp_abs_acf_mean_1_5`：SP500絶対リターン
- `rate_acf_mean_1_5`：DGS10変化率
- `rate_abs_acf_mean_1_5`：DGS10絶対変化率

削除するのは各系列の`acf1`と`acf5`の計8特徴量である。jointはbaselineと同じ`joint_large_move_95`を使用し、ZA・MF2配分、blocked CV、LightGBMパラメータ、0.5閾値は変更しない。

## 学習・B5評価

| 指標 | lag 1〜5平均版 | baseline |
|---|---:|---:|
| CV AUC平均 | 0.7458 | 0.7657 |
| CV AUC標準偏差 | 0.2896 | 0.2114 |
| CV AUC最小値 | 0.4189 | 0.5426 |
| B5 AUC | 0.9489 | 0.9654 |
| B5 logloss | 0.3764 | 0.3354 |
| B5実データRecall | 0.4941 | 0.8471 |
| B5生成データSpecificity | 0.9800 | 0.9650 |

生成データSpecificityはわずかに上がったが、CV、B5 AUC、logloss、実データRecallはbaselineより悪化した。lagごとの差を平均で潰したことにより、実データを拾う情報が減った可能性がある。
