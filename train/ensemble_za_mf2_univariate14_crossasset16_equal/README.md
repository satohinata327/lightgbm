# 単系列14 + 複数資産間16 等重みアンサンブル

単系列14特徴量モデルと複数資産間16特徴量モデルの実データ確率を0.5ずつ平均するmodel-level fusionである。各LightGBMは再学習せず、個別の木構造を維持する。判定閾値は0.5である。

## 計算式

単系列モデルの出力を `p_univariate14`、複数資産間モデルの出力を `p_crossasset16` とすると、最終確率は次の通りである。

```text
prob_real = 0.5 * p_univariate14 + 0.5 * p_crossasset16
```

`prob_real >= 0.5` のとき実データと判定する。0.5は学習には使わず、確率をラベルに変換する段階だけで使う。

## B5評価

| 指標 | 結果 |
|---|---:|
| AUC | 0.9929 |
| logloss | 0.4694 |
| accuracy | 0.9333 |
| balanced accuracy | 0.8950 |
| 実データRecall | 0.8000 |
| 生成データSpecificity | 0.9900 |

詳細は `results/summary.json` と `results/test_predictions.csv` に保存している。
