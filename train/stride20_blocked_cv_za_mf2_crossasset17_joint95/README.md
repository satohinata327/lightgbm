# ZA + MF2 複数資産間17特徴量モデル（joint 95%あり）

複数資産間16特徴量モデルに、baselineと同じ`joint_large_move_95`だけを追加した対照実験である。

`joint_large_move_95`は、SP500とDGS10の絶対変化が同じ日にそれぞれの95%分位点を超えた日数を、`window_length=1260`で割った同時大変動率である。

学習データ、ZA・MF2のサンプル割り当て、blocked CV、LightGBMパラメータは`stride20_blocked_cv_za_mf2_crossasset16`と同一であり、差はこの1特徴量だけである。

```bash
.venv/bin/python common/train_blocked_cv.py \
  --config train/stride20_blocked_cv_za_mf2_crossasset17_joint95/config.json
```

## 学習結果

| 指標 | crossasset16 | crossasset17 + joint95 |
|---|---:|---:|
| CV平均AUC | 0.7051 | 0.8698 |
| CV最小AUC | 0.4927 | 0.7045 |
| B5 AUC | 0.9940 | 0.9916 |
| B5 logloss | 0.4093 | 0.3595 |
| B5実データRecall | 0.7882 | 0.7765 |
| B5生成データSpecificity | 0.9800 | 0.9750 |

joint追加によってblocked CVの順位性能とB5 loglossは改善した。一方、B5 AUC、Recall、Specificityはわずかに低下した。
