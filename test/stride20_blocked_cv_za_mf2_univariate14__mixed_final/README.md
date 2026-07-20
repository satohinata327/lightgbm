# 単資産14特徴量モデルによるmixed final評価

`stride20_blocked_cv_za_mf2_univariate14`を`mixed_final_masked.csv`へ適用した結果である。mixed finalは学習、early stopping、モデル選択に使用していない。

実データ候補として重視しているmask3・4・6・7の確率は、mask3が0.6477、mask4が0.3895、mask6が0.6815、mask7が0.6133だった。mask4を上位へ配置できず、mask12が0.5942、mask10が0.5801、mask8が0.5602となったため、単資産特徴量だけではbaselineの判別結果を再現できていない。
