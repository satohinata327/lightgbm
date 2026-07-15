# Stride 20 blocked CV：ZA 100件＋MF2 100件

5つの各blockにはZA 100系列とMF2 100系列を含める。固定seedで各ソースを独立にシャッフルし、生成サンプルをblock間で再利用しない。実データblock、特徴量、LightGBMパラメータ、early stopping設定は`stride20_blocked_cv_za_final7`から変更していない。

## Blocked cross-validation結果

| Fold | Validation期間 | AUC | Balanced accuracy | 実データRecall |
|---|---|---:|---:|---:|
| 1 | 1978-1990 | 0.5426 | 0.4975 | 0.0000 |
| 2 | 1990-2001 | 0.7915 | 0.8387 | 0.6824 |
| 3 | 2001-2013 | 0.9631 | 0.4900 | 0.0000 |

- 平均AUC：0.7657
- AUC標準偏差：0.2114
- 最低AUC：0.5426
- 最終的な木の本数：36

ZAのみのrunと比較してCVの平均AUCと最低AUCは上昇したが、fold間のばらつきと固定閾値における実データRecallの問題は残っている。

## 最終B5 test

- AUC: 0.9654
- Balanced accuracy：0.9060
- 実データRecall：0.8471
- 生成データ全体のSpecificity：0.9650
- ZA Specificity：0.9300
- MF2 Specificity：1.0000

B5 AUCと実データRecallはZAのみのrunより低い。複数ソースモデルはholdoutしたMF2サンプルを0.5閾値で完全に分離したが、B5の実データ85窓中13窓を生成データと判定した。

## Mixed final

実データ確率の上位4つはmask3（0.8245）、mask6（0.8050）、mask7（0.7468）、mask4（0.6895）である。0.5を超えたのはこの4つだけだった。mixed finalファイルは学習、early stopping、モデル選択には使用していない。
