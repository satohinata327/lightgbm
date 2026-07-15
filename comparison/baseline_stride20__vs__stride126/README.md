# LightGBM stride比較

この対照実験では、実データ窓のstrideだけを`20`から`126`へ変更した。入力ファイル、26特徴量、分割比率、`StandardScaler`、LightGBMハイパーパラメータ、乱数seed、early stoppingは変更していない。

## Test結果

| 実験 | AUC | 正解率 | Balanced accuracy | Logloss | 実データRecall | 生成データSpecificity |
|---|---:|---:|---:|---:|---:|---:|
| stride 20 | 0.8021 | 0.5910 | 0.4950 | 0.6758 | 0.0000 | 0.9900 |
| stride 126 | 0.7516 | 0.8874 | 0.4925 | 0.2907 | 0.0000 | 0.9850 |

stride 126の通常の正解率が高いのは、testデータのクラス不均衡がより大きいためである（実データ22件、生成データ200件）。どちらのモデルも、既定の0.5閾値ではtestの実データ窓をすべて生成データと判定した。そのため、Balanced accuracyと実データRecallには改善がない。
