# ZA + MF2 複数資産間11特徴量モデル（smooth joint）

現在の11特徴量モデルの`joint_large_move_95`を、連続的な`joint_large_move_rank95_smooth_tau02`へ置き換えた対照実験である。他の10特徴量と学習条件は同一である。

各系列の絶対変化を窓内の経験分位順位`u`へ変換し、95%点を中心、幅2パーセンタイルのシグモイドでtail weightを計算する。

```text
w(u) = sigmoid((u - 0.95) / 0.02)
smooth_joint = mean(w(u_sp) * w(u_dgs))
```

95%を超えた日だけを0/1で数えず、95%点に近い日と超過の強さも連続的に反映する。

## 学習・B5評価結果

| 指標 | smooth joint版 | hard joint版 |
|---|---:|---:|
| CV AUC平均 | 0.8403 | 0.7710 |
| CV AUC標準偏差 | 0.0536 | 0.1225 |
| CV AUC最小値 | 0.7786 | 0.6331 |
| B5 AUC | 0.9934 | 0.9836 |
| B5 logloss | 0.3775 | 0.4049 |
| B5 実データRecall | 0.9176 | 0.9176 |
| B5 生成データSpecificity | 0.9500 | 0.9450 |

CVとB5ではhard joint版より良好である。一方、mixed_finalではmask10とmask9も0.5を超えたため、目的の4マスクだけを分離する性能はhard joint版の方が高い。このモデルは、境界の滑らかさと汎化評価を改善した対照実験として扱う。
