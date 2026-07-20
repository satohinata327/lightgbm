# Stride 20 blocked CV：ZA＋MF2 Optuna v1

26特徴量の`stride20_blocked_cv_za_mf2`を基準とし、データ、blocked CV、ZA・MF2のサンプル割当、目的関数、early stoppingを固定してLightGBMのハイパーパラメータだけをOptunaで探索した。

## 探索条件

- 試行数：50
- sampler：TPE（seed 42）
- LightGBM objective：`binary`
- early stopping：`binary_logloss`、50 rounds
- Optuna objective：`CV平均AUC - 0.25 × CV AUC標準偏差`
- 探索中にB5とmixed finalは使用していない

## 最良trial

- Trial：42
- Robust AUC：0.7925
- CV平均AUC：0.8108
- CV AUC標準偏差：0.0732
- CV最低AUC：0.7655

| パラメータ | 値 |
|---|---:|
| `learning_rate` | 0.05973 |
| `max_depth` | 3 |
| `num_leaves` | 7 |
| `min_child_samples` | 70 |
| `colsample_bytree` | 0.6 |
| `subsample` | 0.6 |
| `subsample_freq` | 6 |
| `reg_alpha` | 1.70882 |
| `reg_lambda` | 0.66437 |
| `min_split_gain` | 0.76193 |

Optunaが提案した`num_leaves`は19だったが、`max_depth=3`との整合性を保つ制約により実効値は`2^3-1=7`である。

## Fold結果

| Fold | AUC | Balanced accuracy | 実データRecall | Best iteration |
|---|---:|---:|---:|---:|
| 1 | 0.7655 | 0.4975 | 0.0000 | 15 |
| 2 | 0.7717 | 0.8118 | 0.6235 | 36 |
| 3 | 0.8952 | 0.4975 | 0.0000 | 62 |

最終モデルの木の本数にはbest iterationの中央値である36を使用した。AUCの時代間安定性は改善した一方、0.5閾値での実データRecallはFold 1とFold 3で0であり、確率校正と閾値性能は改善していない。

## B5最終評価

| 指標 | Baseline | Optuna v1 |
|---|---:|---:|
| AUC | 0.9654 | 0.9185 |
| Balanced accuracy | 0.9060 | 0.8868 |
| 実データRecall | 0.8471 | 0.8235 |
| 生成データSpecificity | 0.9650 | 0.9500 |

生成元別のspecificityはZA 0.91、MF2 0.99だった。CV平均と安定性は向上したが、完全に未使用だったB5ではbaselineを下回った。

## Mixed final

上位はmask3（0.8741）、mask6（0.8205）、mask7（0.6797）、mask4（0.5577）となり、目的としている4件が上位4件を占めた。ただしmask1も0.5284で0.5を超えたため、0.5閾値では偽陽性が1件ある。

baselineもmask3・4・6・7を上位4件として分離し、mask1は0.4384だった。そのためmixed finalだけを見ると、Optunaモデルがbaselineより明確に優れているとはいえない。

## 判断

Optuna v1は、浅い木、強めの正則化、大きい`min_child_samples`を選択し、CV AUCの時代間ばらつきを抑えた。しかしB5性能は低下したため、baselineを置き換える決定的な改善ではない。独立評価を重視する場合、引き続き26特徴量baselineを主モデルとするのが妥当である。
