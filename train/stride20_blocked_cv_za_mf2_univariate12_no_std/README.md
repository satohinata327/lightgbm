# ZA + MF2 単系列12特徴量（標準偏差なし）モデル

単系列14特徴量モデルから `sp_std` と `rate_std` だけを除いた対照実験である。実データblock、ZA・MF2の生成データ割り当て、LightGBMパラメータなど、特徴量以外の条件は単系列14特徴量モデルと同一である。

使用する特徴量は、SP500とDGS10それぞれの平均、歪度、超過尖度、絶対リターン自己相関、および将来ボラティリティとの関係である。SP500についてはレバレッジ相関も使用する。

```bash
.venv/bin/python common/train_blocked_cv.py \
  --config train/stride20_blocked_cv_za_mf2_univariate12_no_std/config.json
```
