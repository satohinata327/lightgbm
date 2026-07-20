# Stride 20 blocked CV：ZA 100件のみ

ZA・MF2・SigGAN均等モデルとの比較用に、ZAだけを各block 100件使用したモデルである。既存の`stride20_blocked_cv_za_final7`はZAを各block 200件使用しているため、別実験として作成した。

`window_length=1260`、`stride=20`、26特徴量、実データblock、乱数seed、LightGBMパラメータ、early stoppingはZA＋MF2 baselineと同じである。

## Blocked cross-validation

| Fold | AUC | Balanced accuracy | 実データRecall | Best iteration |
|---|---:|---:|---:|---:|
| 1 | 0.5539 | 0.4950 | 0.0000 | 8 |
| 2 | 0.8173 | 0.6529 | 0.3059 | 37 |
| 3 | 0.9250 | 0.9350 | 1.0000 | 1 |

- 平均AUC：0.7654
- AUC標準偏差：0.1909
- 最低AUC：0.5539
- 最終的な木の本数：8

## 最終B5 test

- AUC：0.9866
- Balanced accuracy：0.9565
- 実データRecall：0.9529
- ZA Specificity：0.9600
- Logloss：0.5719

B5の順位性能と0.5閾値の分類性能は高い。ただし最終モデルは8本の木しかなく、確率が0.5付近へ集中しているため、確率値の余裕は小さい。

## Mixed final

| 順位 | mask | 実データ確率 |
|---:|---|---:|
| 1 | mask3 | 0.6032 |
| 2 | mask6 | 0.5983 |
| 3 | mask4 | 0.5691 |
| 4 | mask10 | 0.5647 |
| 5 | mask1 | 0.5582 |
| 6 | mask8 | 0.5432 |
| 7 | mask7 | 0.5310 |

目的のmask3・4・6・7はすべて0.5を超えたが、mask10・1・8も実データと判定した。したがって、B5性能は高いものの、mixed_finalで目的の4件だけを分離する性能はZA＋MF2 baselineより劣る。

Mixed finalは学習、early stopping、モデル選択には使用していない。
