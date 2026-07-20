# Stride 20 blocked CV：ZA＋MF2＋60日rolling correlation（5特徴量）

`stride20_blocked_cv_za_mf2`を対照モデルとし、従来の26特徴量へ60日rolling correlationの分布特徴5個を追加した。実データblock、`window_length=1260`、`stride=20`、ZA・MF2の系列割当、LightGBMパラメータ、early stopping、乱数seedは同一である。

追加特徴量：

- `rolling_corr_60_mean`
- `rolling_corr_60_std`
- `rolling_corr_60_q10`
- `rolling_corr_60_q50`
- `rolling_corr_60_q90`

極端値である`min`・`max`と、`negative_fraction`は使用していない。合計特徴量数は31である。

## Blocked cross-validation結果

| Fold | Validation期間 | AUC | Balanced accuracy | 実データRecall |
|---|---|---:|---:|---:|
| 1 | 1978〜1990年 | 0.6504 | 0.4975 | 0.0000 |
| 2 | 1990〜2001年 | 0.7618 | 0.8362 | 0.6824 |
| 3 | 2001〜2013年 | 0.2636 | 0.4875 | 0.0000 |

- 平均AUC：0.5586
- AUC標準偏差：0.2615
- 最低AUC：0.2636
- Best iteration：17、50、15
- 最終的な木の本数：17

## 最終B5 test

- AUC：0.9218
- Balanced accuracy：0.6816
- 実データRecall：0.3882
- 生成データSpecificity：0.9750
- ZA Specificity：0.9500
- MF2 Specificity：1.0000

## 現行26特徴量モデルとの比較

| 指標 | 現行26特徴量 | ＋rolling correlation 5特徴量 |
|---|---:|---:|
| CV平均AUC | 0.7657 | 0.5586 |
| CV AUC標準偏差 | 0.2114 | 0.2615 |
| CV最低AUC | 0.5426 | 0.2636 |
| B5 AUC | 0.9654 | 0.9218 |
| B5 Balanced accuracy | 0.9060 | 0.6816 |
| B5実データRecall | 0.8471 | 0.3882 |

極端値を除いても主要評価は現行26特徴量モデルより悪化した。

## 特徴量gainの診断

上位は`sp_acf1`（28.9%）、`rolling_corr_60_q10`（21.3%）、`joint_large_move_95`（13.8%）、`corr_sp_rate`（10.5%）、`sp_mean`（7.4%）だった。極端な最小値への依存は除けたが、rolling correlationの下側分位点への依存が残っている。

## Mixed final

mask6（0.6851）、mask3（0.6693）、mask7（0.5114）、mask1（0.5097）が0.5を超えた。mask4は0.4416であり、目的であるmask3・4・6・7とその他の分離は実現していない。

Mixed finalは学習、early stopping、モデル選定には使用していない。旧8特徴量版のグラフだけは、比較用に`prob_real_ranking_rolling_corr60_8features.svg`として保持している。
