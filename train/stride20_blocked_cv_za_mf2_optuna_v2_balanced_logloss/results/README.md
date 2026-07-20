# ZA＋MF2 Optuna v2：Balanced logloss

26特徴量の`stride20_blocked_cv_za_mf2`を基準とし、データ、blocked CV、生成データ割当、LightGBMの目的関数とearly stoppingを変更せず、Optuna側の目的関数だけをbalanced loglossへ変更した。

## 目的関数

各Foldで実データと生成データのloglossを別々に計算する。

```text
balanced_logloss = (real_logloss + generated_logloss) / 2
Optuna objective = 3 Fold平均balanced_logloss
                   + 0.5 × 最悪Fold balanced_logloss
```

Optunaはこの値を最小化する。LightGBMは従来どおり`objective=binary`、early stoppingは`binary_logloss`である。

## 探索

- 総trial数：50
- Trial 0：baselineパラメータ
- sampler：TPE、seed 42
- B5とmixed finalは探索・モデル選択に未使用

| 候補 | 目的値 | 平均balanced logloss | 最悪Fold balanced logloss |
|---|---:|---:|---:|
| Baseline Trial 0 | 1.0498 | 0.6618 | 0.7760 |
| 最良Trial 39 | 1.0063 | 0.6123 | 0.7881 |

目的値は約4.1%改善した。平均損失は改善したが、最悪Fold損失はわずかに悪化しており、平均側の改善によって選ばれた結果である。

## 最良パラメータ

| パラメータ | 値 |
|---|---:|
| `learning_rate` | 0.05779 |
| `max_depth` | 4 |
| `num_leaves` | 15 |
| `min_child_samples` | 20 |
| `colsample_bytree` | 1.0 |
| `subsample` | 0.6 |
| `subsample_freq` | 4 |
| `reg_alpha` | 0.0 |
| `reg_lambda` | 0.0 |
| `min_split_gain` | 0.95497 |
| 最終的な木の本数 | 14 |

## Fold結果

| Fold | AUC | Balanced logloss | Balanced accuracy | 実データRecall |
|---|---:|---:|---:|---:|
| 1 | 0.4975 | 0.7881 | 0.4975 | 0.0000 |
| 2 | 0.8596 | 0.4995 | 0.8538 | 0.7176 |
| 3 | 0.9739 | 0.5491 | 0.6076 | 0.2353 |

Fold 3のRecallはbaselineの0から0.2353へ改善した。一方、Fold 1はRecall 0のままで、AUCも0.5をわずかに下回った。

## B5評価

| 指標 | Baseline | Optuna v1 AUC | Optuna v2 balanced logloss |
|---|---:|---:|---:|
| AUC | 0.9654 | 0.9185 | 0.9251 |
| 通常logloss | 0.3354 | 0.3610 | 0.3707 |
| Balanced logloss | 0.3924 | 0.4504 | 0.3955 |
| Balanced accuracy | 0.9060 | 0.8868 | 0.9188 |
| 実データRecall | 0.8471 | 0.8235 | 0.9176 |
| 生成データSpecificity | 0.9650 | 0.9500 | 0.9200 |

v2は実データRecallとbalanced accuracyを改善した。一方で生成データの誤検出が増え、AUC、通常logloss、balanced loglossはbaselineを超えていない。生成元別specificityはZA 0.86、MF2 0.98で、主にZAの偽陽性が増えた。

## Mixed final

上位4件はmask3（0.7805）、mask6（0.7535）、mask7（0.7535）、mask4（0.7399）で、目的の4件が上位を占めた。ただしmask1も0.5618で0.5を超えた。baselineではmask1が0.4384だったため、4件とそれ以外の分離はbaselineの方が明確である。

## 判断

目的関数の変更により、実データを0.5以上へ押し上げる方向には改善し、B5実データRecallは最も高くなった。しかし生成データも実データ判定しやすくなり、最終的な分離性能ではbaselineを置き換えられない。現時点の主モデルは引き続きZA＋MF2 baselineが妥当である。
