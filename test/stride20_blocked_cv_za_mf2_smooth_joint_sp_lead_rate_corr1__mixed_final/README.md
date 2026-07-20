# smooth joint＋SP500リード相関：mixed_final評価

| 順位 | mask | 実データ確率 |
|---:|---|---:|
| 1 | mask3 | 0.7734 |
| 2 | mask6 | 0.7716 |
| 3 | mask7 | 0.7264 |
| 4 | mask10 | 0.6682 |
| 5 | mask4 | 0.6646 |
| 6 | mask1 | 0.6416 |

`sp_lead_rate_corr_1`のSHAP寄与はmask4で+0.3509、mask10で-0.0727となり、追加特徴量単体は意図した方向に働いた。しかしsmooth jointの寄与がmask4で+0.5761、mask10で+0.7807となり、mask10をより強く押し上げた。最終確率ではmask10がmask4を0.0036上回り、目的の分離は達成していない。
