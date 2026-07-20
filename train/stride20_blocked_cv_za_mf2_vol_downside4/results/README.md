# Stride 20 blocked CV：ZA＋MF2、依存構造4特徴量追加

baselineの26特徴量、ZA 100件＋MF2 100件、実データblock、乱数seed、LightGBMパラメータを固定し、次の4特徴量だけを追加した30特徴量モデルである。

- `vol_abs_corr_coupling_60`：60日統合ボラティリティと60日相関の絶対値の相関
- `high_vol_abs_corr_gap_60`：統合ボラティリティ上位20%期とそれ以外の期間における、60日相関の絶対値の平均差
- `sp_downside_corr_gap_10`：SP500下位10%日と中央20〜80%日におけるSP500・DGS10相関の差
- `sp_downside_abs_corr_gap_10`：上記2相関の絶対値の差

ここでDGS10は債券価格ではなく、元CSVに格納された金利変化として扱う。

## Blocked cross-validation

| Fold | AUC | Balanced accuracy | 実データRecall | best iteration |
|---|---:|---:|---:|---:|
| 1 | 0.3558 | 0.4975 | 0.0000 | 17 |
| 2 | 0.7764 | 0.8387 | 0.6824 | 59 |
| 3 | 0.4479 | 0.4700 | 0.0000 | 13 |

- 平均AUC：0.5267（baseline：0.7657）
- AUC標準偏差：0.2211（baseline：0.2114）
- 最終的な木の本数：17（baseline：36）

## 最終B5 test

| 指標 | 4特徴量追加 | baseline |
|---|---:|---:|
| AUC | 0.9383 | 0.9654 |
| Balanced accuracy | 0.8675 | 0.9060 |
| 実データRecall | 0.8000 | 0.8471 |
| 生成データSpecificity | 0.9350 | 0.9650 |
| Logloss | 0.4702 | 0.3354 |

すべての主要指標がbaselineより悪化した。

## Mixed final

実データ確率の上位はmask3（0.7014）、mask6（0.6820）、mask7（0.6787）、mask4（0.6246）、mask1（0.5680）だった。目的としているmask3・4・6・7はすべて0.5を超えたが、baselineでは0.5未満だったmask1も実データと判定した。

最終モデルのgainでは追加特徴量のうち`high_vol_abs_corr_gap_60`だけが明確に使用された。下落条件付きの2特徴量は最終モデルの分岐にほぼ使用されていない。したがって、この4特徴量をまとめて追加する構成はbaselineの改善とは判断しない。
