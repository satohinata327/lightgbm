# ZA + MF2 hybrid 18特徴量モデル

単資産14特徴量モデルと複数資産間16特徴量モデルから、学習時gain、未使用B5、低重複pseudo finalを基準に18特徴量を選んだ統合モデルである。mixed finalは特徴量選定に使用していない。

単資産側ではSP500の標準偏差、歪度、超過尖度、絶対リターンACF、レバレッジ相関と、DGS10の標準偏差、絶対変化ACF、翌5日間実現ボラティリティとの相関を使う。資産間側では双方向リード・ラグ、60日rolling correlation、絶対変動相関、20日実現ボラティリティ相関を使う。

実データblock、生成系列の割り当て、LightGBMパラメータはbaselineと同一である。

```bash
.venv/bin/python common/train_blocked_cv.py \
  --config train/stride20_blocked_cv_za_mf2_hybrid18/config.json
```
