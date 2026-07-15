# LightGBM可視化

LightGBMモデルの構造や判定根拠を調べるコードと成果物を保存する。

## Gain上位の木

`visualize_top_gain_trees.py`は各木に含まれる`split_gain`の合計を計算し、上位の木を個別のSVGとして出力する。Graphvizやpydotplusには依存しない。

```bash
lightgbm/.venv/bin/python lightgbm/visualize/visualize_top_gain_trees.py
```

既定の出力先：

```text
lightgbm/visualize/stride20_blocked_cv_za_mf2/top_gain_trees/
```

各分岐には標準化後の閾値と、`StandardScaler`を逆変換した元尺度の閾値を併記する。
