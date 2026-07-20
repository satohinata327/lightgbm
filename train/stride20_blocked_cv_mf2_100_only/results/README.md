# Stride 20 blocked CV：MF2 100件のみ

ZA 100-onlyおよびZA＋MF2 baselineとの比較用に、MF2だけを各block 100件使用したモデルである。

`window_length=1260`、`stride=20`、26特徴量、実データblock、LightGBMパラメータ、early stoppingはZA＋MF2 baselineと同じである。MF2の抽出には実効seed 43を使い、ZA＋MF2 baselineで使われたMF2系列とblock割当を一致させている。

## Blocked cross-validation

| Fold | AUC | Balanced accuracy | 実データRecall | Best iteration |
|---|---:|---:|---:|---:|
| 1 | 0.3687 | 0.5000 | 0.0000 | 5 |
| 2 | 0.9606 | 0.8465 | 0.7529 | 63 |
| 3 | 0.9881 | 0.9841 | 0.9882 | 33 |

- 平均AUC：0.7725
- AUC標準偏差：0.3499
- 最低AUC：0.3687
- 最終的な木の本数：33

Fold 2とFold 3は良好だが、最も古い時代から次の時代へ移るFold 1のAUCとRecallが低い。時代間の汎化性能は不安定である。

## 最終B5 test

- AUC：0.9449
- Balanced accuracy：0.8529
- 実データRecall：0.7059
- MF2 Specificity：1.0000
- Logloss：0.3887

## Mixed final

| 順位 | mask | 実データ確率 |
|---:|---|---:|
| 1 | mask3 | 0.8146 |
| 2 | mask6 | 0.8135 |
| 3 | mask7 | 0.7631 |
| 4 | mask9 | 0.5464 |
| 5 | mask1 | 0.4863 |
| 6 | mask4 | 0.4773 |

0.5を超えたのはmask3・6・7・9である。目的のmask4は0.4773で生成データ側となり、代わりにmask9を実データと誤判定した。MF2-onlyでは目的の4件を分離できておらず、ZAを加えたbaselineの方が明確に良い。

Mixed finalは学習、early stopping、モデル選択には使用していない。
