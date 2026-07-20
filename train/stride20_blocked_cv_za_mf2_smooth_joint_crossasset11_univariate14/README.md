# smooth joint複数資産間11特徴量＋単系列14特徴量モデル

複数資産間11特徴量モデルに単系列14特徴量を追加した、合計25特徴量のLightGBMモデルである。`joint`には離散的な`joint_large_move_95`ではなく、連続的な`joint_large_move_rank95_smooth_tau02`を使用する。

## 条件

- 正例：実データのB1〜B4
- 負例：ZA 100系列＋MF2 100系列（各ブロック）
- `window_length=1260`
- `stride=20`
- B5は最終テスト専用
- LightGBMパラメータは従来baselineと同一
- 判定閾値：0.62

判定閾値0.62はmixed_finalにおける4位mask4（0.6324）と5位mask10（0.6007）の間から選んだ。したがって、これはmixed_finalへの適合を優先した閾値であり、未知データに対する事前固定閾値ではない。

## 評価

| 指標 | 値 |
|---|---:|
| CV AUC平均 | 0.6043 |
| CV AUC標準偏差 | 0.3101 |
| B5 AUC | 0.9908 |
| B5 logloss | 0.4219 |
| B5実データRecall（0.5） | 0.9176 |
| B5生成データSpecificity（0.5） | 0.9700 |

CVはsmooth joint 11特徴量版より悪化している。一方、mixed_finalでは上位4件が目的のmask3・4・6・7となった。
