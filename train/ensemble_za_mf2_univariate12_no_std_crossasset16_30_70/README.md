# 単系列12（標準偏差なし）+ 複数資産間16 0.3–0.7アンサンブル

`sp_std`と`rate_std`を除いた単系列12特徴量モデルを0.3、複数資産間16特徴量モデルを0.7の重みでsoft votingするmodel-level fusionである。

```text
prob_real = 0.3 * p_univariate12_no_std + 0.7 * p_crossasset16
```

判定閾値は0.5とし、各構成モデルは独立した木構造を維持する。

## B5評価

| 指標 | 結果 |
|---|---:|
| AUC | 0.9944 |
| logloss | 0.4440 |
| accuracy | 0.9193 |
| balanced accuracy | 0.8816 |
| 実データRecall | 0.7882 |
| 生成データSpecificity | 0.9750 |

詳細は `results/summary.json` と `results/test_predictions.csv` に保存している。
