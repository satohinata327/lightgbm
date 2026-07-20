# ZA + MF2 単資産14特徴量モデル

`stride20_blocked_cv_za_mf2`を対照として、特徴量だけを単資産のstylized factsに限定した実験である。実データblock、生成系列の割り当て、LightGBMパラメータはbaselineと同一である。

SP500では平均、標準偏差、歪度、超過尖度、絶対リターンACF（lag 1）、当日の絶対リターンと翌5日間の実現ボラティリティの相関、レバレッジ相関（1日先・5日先）を使用する。DGS10では平均、標準偏差、歪度、超過尖度、絶対変化ACF（lag 1）、当日の絶対変化と翌5日間の実現ボラティリティの相関を使用する。

レバレッジ相関は、当日のSP500リターンと、その後の実現ボラティリティとの相関である。負の値ほど、下落後にボラティリティが上昇しやすいことを表す。

```bash
.venv/bin/python common/train_blocked_cv.py \
  --config train/stride20_blocked_cv_za_mf2_univariate14/config.json
```
