# Stride 20 blocked CV：ZA＋MF2＋SigGAN 1000、均等配分

旧`stride20_blocked_cv_za_mf2_sigwgan`実験は残し、新しいSigGANデータ`synthetic_returns_siggan_1000paths_wide1.csv`を使用して学習し直した。

各blockには重複しない生成系列を次のように割り当てた。

- ZA：100件
- MF2：100件
- SigGAN：100件
- 合計：300件

`window_length=1260`、`stride=20`、26特徴量、実データblock、乱数seed、LightGBMパラメータ、early stoppingはZA＋MF2 baselineと同じである。生成負例の合計数はbaselineの200件から300件へ増えている。

## Blocked cross-validation

| Fold | AUC | Balanced accuracy | 実データRecall | Best iteration |
|---|---:|---:|---:|---:|
| 1 | 0.5309 | 0.4967 | 0.0000 | 27 |
| 2 | 0.7712 | 0.8202 | 0.6471 | 67 |
| 3 | 0.3000 | 0.4900 | 0.0000 | 29 |

- 平均AUC：0.5340
- AUC標準偏差：0.2356
- 最低AUC：0.3000
- 最終的な木の本数：29

## 最終B5 test

- AUC：0.8585
- Balanced accuracy：0.8539
- 実データRecall：0.7412
- 生成データ全体のSpecificity：0.9667
- ZA Specificity：0.9100
- MF2 Specificity：0.9900
- SigGAN Specificity：1.0000
- Logloss：0.3854

SigGANは100件すべてを生成データと判別した。一方、実データRecallとAUCはZA＋MF2 baselineより低下した。SigGANの判別が容易であるため、100件への増量が実データと難しい生成データを区別する性能の改善にはつながっていない可能性がある。

## Mixed final

| 順位 | mask | 実データ確率 |
|---:|---|---:|
| 1 | mask6 | 0.7620 |
| 2 | mask3 | 0.7521 |
| 3 | mask4 | 0.6786 |
| 4 | mask7 | 0.5958 |
| 5 | mask1 | 0.5125 |

mask3・4・6・7は0.5を超えたが、mask1も0.5をわずかに超えて誤検出された。したがって、ユーザーが求めるmixed_finalの分離結果についてもZA＋MF2 baselineを超えていない。

Mixed finalは学習、early stopping、モデル選択には使用していない。
