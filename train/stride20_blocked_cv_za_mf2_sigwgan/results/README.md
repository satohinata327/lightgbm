# Stride 20 blocked CV：ZA＋MF2＋SIG-WGAN

直前の`stride20_blocked_cv_za_mf2`を対照モデルとし、生成負例の構成だけを変更した。実データblock、`window_length=1260`、`stride=20`、26特徴量、LightGBMパラメータ、early stopping、乱数seed、各blockの生成負例総数200件は同一である。

各blockの生成負例は次の通りである。

- ZA：88件
- MF2：88件
- SIG-WGAN：24件
- 合計：200件

SIG-WGAN 120件を5 blockへ24件ずつ重複なく割り当てた。ZAとMF2は旧モデルの各block 100件の先頭88件を維持している。

## Blocked cross-validation結果

| Fold | Validation期間 | AUC | Balanced accuracy | 実データRecall |
|---|---|---:|---:|---:|
| 1 | 1978〜1990年 | 0.5656 | 0.4975 | 0.0000 |
| 2 | 1990〜2001年 | 0.8233 | 0.8563 | 0.7176 |
| 3 | 2001〜2013年 | 0.8602 | 0.4900 | 0.0000 |

- 平均AUC：0.7497
- AUC標準偏差：0.1605
- 最低AUC：0.5656
- Best iteration：18、50、18
- 最終的な木の本数：18

## 最終B5 test

- AUC：0.9611
- Balanced accuracy：0.8985
- 実データRecall：0.8471
- 生成データ全体のSpecificity：0.9500
- ZA Specificity：0.9318
- MF2 Specificity：1.0000
- SIG-WGAN Specificity：0.8333

## 直前モデルとの比較

| 指標 | ZA＋MF2 | ZA＋MF2＋SIG-WGAN |
|---|---:|---:|
| CV平均AUC | 0.7657 | 0.7497 |
| CV AUC標準偏差 | 0.2114 | 0.1605 |
| CV最低AUC | 0.5426 | 0.5656 |
| B5 AUC | 0.9654 | 0.9611 |
| B5 Balanced accuracy | 0.9060 | 0.8985 |
| B5実データRecall | 0.8471 | 0.8471 |
| B5生成データSpecificity | 0.9650 | 0.9500 |

SIG-WGAN追加後はCV平均AUCとB5性能がわずかに低下した。一方、CVの標準偏差と最低AUCは改善し、時代間の順位性能はやや安定した。B5のSIG-WGAN 24件中20件を生成データと判定し、4件を実データと誤判定した。

## Mixed final

mask3（0.7059）、mask6（0.6872）、mask7（0.6671）、mask4（0.6294）だけが0.5を超え、今回も上位4件を維持した。ただしmask4と次点mask1（0.4634）の差は0.1660で、ZA＋MF2モデルの約0.2511より小さい。

Mixed finalは学習、early stopping、モデル選定には使用していない。
