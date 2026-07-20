# ZA + MF2 複数資産間10特徴量モデル（jointなし）

現在の11特徴量モデルから`joint_large_move_95`だけを削除した対照実験である。以下は引き続き使用しない。

- `corr_abs_sp_rate`
- `rate_lead_sp_corr_1`
- `rolling_corr60_std/q10/q50/q90`

残りの10特徴量、学習データ、生成データ割り当て、blocked CV、LightGBMパラメータは11特徴量版と同一である。

## 学習結果

| 指標 | jointあり11特徴量 | jointなし10特徴量 |
|---|---:|---:|
| CV平均AUC | 0.7710 | 0.5207 |
| CV最小AUC | 0.6331 | 0.4060 |
| B5 AUC | 0.9836 | 0.9626 |
| B5 logloss | 0.4049 | 0.5011 |
| B5実データRecall | 0.9176 | 0.9412 |
| B5生成データSpecificity | 0.9450 | 0.8800 |

joint削除後は実データRecallが少し上がったが、CV安定性、B5順位性能、logloss、生成データSpecificityは悪化した。
