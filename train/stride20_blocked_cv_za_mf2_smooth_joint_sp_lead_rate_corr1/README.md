# baseline smooth joint＋SP500リード相関モデル

26特徴量baseline smooth joint版へ`sp_lead_rate_corr_1`だけを追加した27特徴量の対照実験である。他の特徴量、ZA・MF2の配分、blocked CV、LightGBMパラメータ、判定閾値0.5は変更しない。

`sp_lead_rate_corr_1`は次の相関を表す。

```text
corr(SP500_t, DGS10_{t+1})
```

## 評価結果

| 指標 | 27特徴量版 | 26特徴量smooth版 |
|---|---:|---:|
| CV AUC平均 | 0.7446 | 0.8245 |
| CV AUC標準偏差 | 0.2700 | 0.2106 |
| B5 AUC | 0.9931 | 0.9780 |
| B5 logloss | 0.3456 | 0.2526 |
| B5実データRecall | 0.9176 | 0.8941 |
| B5生成データSpecificity | 0.9650 | 0.9750 |

B5 AUCとRecallは改善したが、CV AUCとloglossは悪化した。追加特徴量が全期間で安定して有効とは言えない。
