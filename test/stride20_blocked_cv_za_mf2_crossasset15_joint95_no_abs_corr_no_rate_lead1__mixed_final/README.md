# crossasset15 joint95 mixed_final評価

`crossasset17_joint95`から`corr_abs_sp_rate`と`rate_lead_sp_corr_1`を削除してmixed_finalを解析した。

| 順位 | mask | 実データ確率 |
|---:|---|---:|
| 1 | mask3 | 0.7202 |
| 2 | mask6 | 0.7070 |
| 3 | mask4 | 0.6156 |
| 4 | mask13 | 0.6086 |
| 5 | mask7 | 0.5474 |
| 6 | mask9 | 0.3136 |
| 10 | mask1 | 0.2717 |
| 12 | mask10 | 0.2622 |

mask1は0.6883から0.2717、mask10は0.6813から0.2622へ下がり、狙った削除効果は得られた。一方でmask13が0.6086へ上がり、mask7より上位になった。mask13は主に`rolling_corr60_q10`、`rolling_corr60_q90`、`rolling_corr60_q50`によって実データ側へ押し上げられている。
