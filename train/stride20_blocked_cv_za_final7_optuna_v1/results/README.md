# Optunaチューニング結果

- Trial数：60
- 選定fold：B1→B2、B1＋B2→B3、B1＋B2＋B3→B4
- 目的値：`validation AUCの平均 - 0.5 * validation AUCの標準偏差`
- 最終holdout：B5（trial選定およびearly stoppingには不使用）
- Mixed final：trial選定には不使用

## 最良trial

Trial 26のrobust AUCは0.671012だった。Fold AUCの平均は0.710147、標準偏差は0.078270、最低値は0.624706だった。

実際に使用した木のパラメータは`../tuning/best_params.json`に記録している。提案された`num_leaves=23`は、`max_depth=3`のため`7`へ制限した。

## 未調整blocked CVモデルとの比較

| 指標 | 未調整 | Optuna |
|---|---:|---:|
| CV AUC平均 | 0.662775 | 0.710147 |
| CV AUC標準偏差 | 0.202580 | 0.078270 |
| CV AUC最低値 | 0.466147 | 0.624706 |
| Robust AUC | 0.561485 | 0.671012 |
| B5 AUC | 0.990559 | 0.995441 |
| B5 Balanced accuracy | 0.982500 | 0.927500 |
| B5生成データSpecificity | 0.965000 | 0.855000 |

Optunaによってvalidation時代間のAUC安定性が改善し、B5 AUCもわずかに向上した。一方、固定した0.5閾値ではB5の偽陽性が増えたため、閾値調整は別の課題として残る。
