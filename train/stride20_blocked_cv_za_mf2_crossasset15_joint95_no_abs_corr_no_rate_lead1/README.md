# ZA + MF2 複数資産間15特徴量モデル

`crossasset17_joint95`から、mask1とmask10を強く実データ側へ押し上げていた以下の2特徴量だけを除いた対照実験である。

- `corr_abs_sp_rate`
- `rate_lead_sp_corr_1`

`joint_large_move_95`を含む残り15特徴量を使用する。学習データ、生成データ割り当て、blocked CV、LightGBMパラメータは`crossasset17_joint95`と同一である。

```bash
.venv/bin/python common/train_blocked_cv.py \
  --config train/stride20_blocked_cv_za_mf2_crossasset15_joint95_no_abs_corr_no_rate_lead1/config.json
```

## 学習結果

| 指標 | crossasset17 joint95 | 2特徴量削除後 |
|---|---:|---:|
| CV平均AUC | 0.8698 | 0.6642 |
| CV最小AUC | 0.7045 | 0.3967 |
| B5 AUC | 0.9916 | 0.9764 |
| B5 logloss | 0.3595 | 0.4303 |
| B5実データRecall | 0.7765 | 0.9294 |
| B5生成データSpecificity | 0.9750 | 0.9550 |

閾値0.5で実データを拾う能力は上がったが、順位性能、確率品質、時代間安定性は低下した。
