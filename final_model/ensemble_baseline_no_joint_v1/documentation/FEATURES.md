# 特徴量

jointありモデルはbaselineの26特徴量を使用する。内容は単系列の分布・自己相関、2系列の同時点相関、同時極端変動、tail ratioである。

jointなしモデルは、次の特徴量だけを除いた25特徴量を使用する。

```text
joint_large_move_95
```

`joint_large_move_95`は、各1260日窓でSP500とDGS10の絶対変化がそれぞれの95%分位点を同日に超えた割合である。

各モデルが実際に使用する列順は、`models/joint/feature_config.json`と`models/no_joint/feature_config.json`を正とする。
