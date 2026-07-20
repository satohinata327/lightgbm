# 最終モデル：jointあり＋なし等重みアンサンブル v1

ZA 100系列＋MF2 100系列で学習した26特徴量baselineと、その`joint_large_move_95`を除いた25特徴量モデルを、等重みsoft votingで統合した最終モデルである。

```text
prob_real = 0.5 * prob_joint + 0.5 * prob_no_joint
pred_label = 1 if prob_real >= 0.5 else 0
```

## ディレクトリ

- `models/joint/`：jointありモデル、scaler、特徴量・学習設定
- `models/no_joint/`：jointなしモデル、scaler、特徴量・学習設定
- `inference/`：最終モデル専用の推論入口
- `evaluation/b5/`：B5評価
- `evaluation/mixed_final/`：mixed_final評価
- `evaluation/pseudo_final_low_overlap_v2/`：低重複pseudo 7セット評価
- `documentation/`：モデル、特徴量、制約の説明

## 推論

`lightgbm/`をカレントディレクトリとして実行する。

```bash
.venv/bin/python final_model/ensemble_baseline_no_joint_v1/inference/predict.py \
  --input data/evaluation_sets/final/mixed_final_masked.csv \
  --output-dir /tmp/final_model_prediction
```

## 主な評価結果

| 評価 | 結果 |
|---|---:|
| B5 AUC | 0.9669 |
| B5 balanced accuracy | 0.9144 |
| B5実データRecall | 0.8588 |
| B5生成データSpecificity | 0.9700 |
| 低重複pseudo平均AUC | 0.9702 |
| pseudo完全上位2件 | 6 / 7 |
| pseudoで生成データが全て0.5未満 | 7 / 7 |

mixed_finalではmask3・4・6・7だけが0.5を超える。
