# ZA + MF2 複数資産間16特徴量モデル

SP500とDGS10の資産間依存だけを使う対照実験である。単資産の平均、標準偏差、歪度、尖度、自己相関などは使用しない。相関の時間変化、下落時の相関非対称性、同時ボラティリティクラスタリング、リード・ラグ、ボラティリティ・スピルオーバーを16特徴量で表現する。

実データblock、生成系列の割り当て、LightGBMパラメータは`stride20_blocked_cv_za_mf2`と同一である。`joint_large_move_95`は使用しない。

```bash
.venv/bin/python common/train_blocked_cv.py \
  --config train/stride20_blocked_cv_za_mf2_crossasset16/config.json
```
