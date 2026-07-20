# crossasset17 + joint95 mixed_final評価

複数資産間16特徴量に`joint_large_move_95`を追加したモデルでmixed_finalを解析した。

| 順位 | mask | 実データ確率 |
|---:|---|---:|
| 1 | mask6 | 0.7445 |
| 2 | mask4 | 0.7421 |
| 3 | mask3 | 0.7392 |
| 4 | mask7 | 0.7202 |
| 5 | mask1 | 0.6883 |
| 6 | mask10 | 0.6813 |

`mask3,4,6,7`が上位4件を占め、crossasset16で`mask4 < mask10`だった順位は改善した。ただし`mask1`と`mask10`も0.5を超えており、閾値0.5では分離できない。
