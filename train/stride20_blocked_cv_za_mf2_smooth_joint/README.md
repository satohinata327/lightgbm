# ZA＋MF2 baseline smooth joint版

26特徴量baselineの`joint_large_move_95`だけを、連続的な`joint_large_move_rank95_smooth_tau02`へ置き換えた対照実験である。学習データ、blocked CV、LightGBMパラメータ、判定閾値0.5はbaselineと同一である。

```text
w(u) = sigmoid((u - 0.95) / 0.02)
smooth_joint = mean(w(u_sp) * w(u_dgs))
```

ここで`u_sp`と`u_dgs`は、それぞれの1260日窓内における絶対変化の経験分位順位である。

## baselineとの比較

| 指標 | smooth joint版 | baseline |
|---|---:|---:|
| CV AUC平均 | 0.8245 | 0.7657 |
| CV AUC標準偏差 | 0.2106 | 0.2114 |
| CV AUC最小値 | 0.5821 | 0.5426 |
| B5 AUC | 0.9780 | 0.9654 |
| B5 logloss | 0.2526 | 0.3354 |
| B5実データRecall | 0.8941 | 0.8471 |
| B5生成データSpecificity | 0.9750 | 0.9650 |

CV平均とB5の全主要指標はbaselineより改善した。
