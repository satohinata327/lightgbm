# Stride 20 blocked CV：za_final7

時系列順に分けた5つの非重複実データblockと、互いに重複しない5つの生成サンプルblockを使用する。拡張型validation foldはB1→B2、B1＋B2→B3、B1＋B2＋B3→B4である。B5は最終test専用として確保する。

## Cross-validation結果

| Fold | Validation期間 | AUC | Balanced accuracy | 実データRecall |
|---|---|---:|---:|---:|
| 1 | 1978-1990 | 0.4661 | 0.5000 | 0.0000 |
| 2 | 1990-2001 | 0.8708 | 0.8228 | 0.6706 |
| 3 | 2001-2013 | 0.6514 | 0.4875 | 0.0000 |

- 平均AUC：0.6628
- AUC標準偏差：0.2026
- 最終モデルに使用したbest iterationの中央値：18

## 最終B5 test

- 期間：2013〜2025年
- AUC: 0.9906
- Balanced accuracy：0.9825
- 実データRecall：1.0000
- 生成データSpecificity：0.9650

最終testが高性能でも、時代間のCV性能が大きく不安定である事実は変わらない。今後パラメータ設定を比較するときは、両方を報告する必要がある。
