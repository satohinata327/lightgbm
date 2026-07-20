# 最終モデル：jointあり＋なし等重みアンサンブル v1

## 1. 概要

このモデルは、1260日分のSP500リターンとDGS10変化率が実データか生成データかを判別する二値分類モデルである。

次の2つのLightGBMを組み合わせる。

1. `joint`：同時極端変動を含む26特徴量モデル
2. `no_joint`：同時極端変動を除いた25特徴量モデル

各モデルが出力する実データ確率を0.5ずつ平均するsoft votingを使用する。

```text
prob_real = 0.5 * prob_joint + 0.5 * prob_no_joint
pred_label = 1 if prob_real >= 0.5 else 0
```

- `prob_real`：アンサンブルが推定した実データ確率
- `pred_label=1`：実データ判定
- `pred_label=0`：生成データ判定

アンサンブル自体を追加学習しているわけではなく、学習済み2モデルの確率平均である。

## 2. 入力データ

1サンプルは次の2系列から構成される。

- SP500の日次リターン
- DGS10の日次変化率
- 各系列の長さ：`window_length=1260`

学習時の実データ窓は`stride=20`で切り出す。mixed形式CSVでは、次のような列名を想定する。

```text
mask1_sp500
mask1_DGS10
mask2_sp500
mask2_DGS10
...
```

## 3. 学習データ

正例を実データ、負例を生成データとする。

| ラベル | 内容 |
|---|---|
| 1 | `train_sp500_us10y.csv`から切り出した実データ窓 |
| 0 | ZA生成系列とMF2生成系列 |

生成データは各blockで次の配分を使用する。

| 生成元 | 系列数 |
|---|---:|
| ZA final7 | 100 |
| MF2 GARCH regime correlation | 100 |
| 合計 | 200 |

ZAは市場シミュレーションによって金融市場の時系列を生成するモデルである。本モデルでは、ZAが生成したSP500・DGS10の疑似系列を負例として使用する。

MF2はGARCHとレジーム依存相関を用いて時系列を生成するモデルである。ZAとMF2では生成方法と再現しやすい統計的性質が異なるため、片方だけに特化した判別器にならないよう両方を負例へ入れている。

ここで「負例」は品質が低いという意味ではない。二値分類において、ラベル0として扱う生成データを指す用語である。同様に「正例」はラベル1の実データを指す。

実データは時系列順にB1〜B5へ分割する。将来期間を過去期間の学習へ混ぜないblocked CVを採用し、B5は学習に使用せず最後のテストだけに使用する。

| Fold | 学習block | Validation block |
|---|---|---|
| 1 | B1 | B2 |
| 2 | B1＋B2 | B3 |
| 3 | B1＋B2＋B3 | B4 |
| 最終学習 | B1＋B2＋B3＋B4 | B5でテスト |

## 4. 使用する特徴量

`sp_`はSP500、`rate_`はDGS10を表す。特徴量は各1260日窓全体から1つの値へ集約する。

### 4.1 分布の中心とばらつき：4特徴量

| 特徴量 | 定義・意味 |
|---|---|
| `sp_mean` | SP500リターンの平均 |
| `sp_std` | SP500リターンの標本標準偏差。ボラティリティの大きさ |
| `rate_mean` | DGS10変化率の平均 |
| `rate_std` | DGS10変化率の標本標準偏差 |

### 4.2 分布の非対称性と裾：4特徴量

| 特徴量 | 定義・意味 |
|---|---|
| `sp_skew` | SP500リターンの歪度。負なら左裾が長い傾向 |
| `sp_kurt` | SP500リターンの超過尖度。0より大きいほど正規分布よりfat tail |
| `rate_skew` | DGS10変化率の歪度 |
| `rate_kurt` | DGS10変化率の超過尖度 |

実装上の尖度は4次標準化モーメントから3を引いた超過尖度である。

### 4.3 絶対変化の分位点：6特徴量

| 特徴量 | 定義・意味 |
|---|---|
| `sp_q90_abs` | SP500絶対リターンの90%分位点 |
| `sp_q95_abs` | SP500絶対リターンの95%分位点 |
| `sp_q99_abs` | SP500絶対リターンの99%分位点 |
| `rate_q90_abs` | DGS10絶対変化率の90%分位点 |
| `rate_q95_abs` | DGS10絶対変化率の95%分位点 |
| `rate_q99_abs` | DGS10絶対変化率の99%分位点 |

符号を除いた変化の大きさについて、通常時から極端時までの裾の形を表す。

### 4.4 リターンの自己相関：4特徴量

| 特徴量 | 定義・意味 |
|---|---|
| `sp_acf1` | SP500リターンのlag 1自己相関 |
| `sp_acf5` | SP500リターンのlag 5自己相関 |
| `rate_acf1` | DGS10変化率のlag 1自己相関 |
| `rate_acf5` | DGS10変化率のlag 5自己相関 |

実データの日次リターンでは強い線形自己相関が生じにくいため、生成系列に不自然な時系列依存がないかを捉える。

### 4.5 絶対変化の自己相関：4特徴量

| 特徴量 | 定義・意味 |
|---|---|
| `sp_abs_acf1` | SP500絶対リターンのlag 1自己相関 |
| `sp_abs_acf5` | SP500絶対リターンのlag 5自己相関 |
| `rate_abs_acf1` | DGS10絶対変化率のlag 1自己相関 |
| `rate_abs_acf5` | DGS10絶対変化率のlag 5自己相関 |

絶対変化の自己相関は、ボラティリティが高い期間・低い期間が持続するボラティリティクラスタリングに対応する。

### 4.6 資産間依存：2特徴量

| 特徴量 | 使用モデル | 定義・意味 |
|---|---|---|
| `corr_sp_rate` | 両方 | 同日のSP500リターンとDGS10変化率のPearson相関 |
| `joint_large_move_95` | jointのみ | 両系列の絶対変化が、それぞれの95%分位点を同日に超えた日数の割合 |

`joint_large_move_95`は次の式で計算する。

```text
joint_large_move_95
= count(|sp_t| > q95_sp and |rate_t| > q95_rate) / 1260
```

この特徴量は同時極端変動を捉える一方、1260日中の数日の差で値が変わる。その感度を補うため、同特徴量を含まないモデルもアンサンブルへ入れている。

### 4.7 標準偏差で正規化した裾：2特徴量

| 特徴量 | 定義・意味 |
|---|---|
| `sp_tail_ratio` | `sp_q99_abs / sp_std` |
| `rate_tail_ratio` | `rate_q99_abs / rate_std` |

単純なボラティリティ水準ではなく、標準的な変動に対して99%分位点がどの程度大きいかを表す。

### 4.8 モデルごとの特徴量数

| モデル | 特徴量数 | 違い |
|---|---:|---|
| joint | 26 | 上記すべてを使用 |
| no_joint | 25 | `joint_large_move_95`だけを除外 |

実際の列名と順序は、次の設定を正とする。

- `models/joint/feature_config.json`
- `models/no_joint/feature_config.json`

## 5. LightGBM

各構成モデルはLightGBMの勾配ブースティング決定木による二値分類器である。複数の決定木を逐次追加し、それまでの予測誤差を後続の木が補正する。

目的関数はbinary loglossである。実データ確率を`p_i`、正解ラベルを`y_i`とすると、最小化する基本形は次のとおりである。

```text
L = -(1/N) * sum[y_i * log(p_i) + (1-y_i) * log(1-p_i)]
```

主要パラメータは両モデルで同じである。

| パラメータ | 値 |
|---|---:|
| `learning_rate` | 0.03 |
| `max_depth` | 4 |
| `num_leaves` | 15 |
| `min_child_samples` | 20 |
| `n_estimators` | 最大1000 |
| `colsample_bytree` | 0.8 |
| `subsample` | 0.8 |
| `class_weight` | balanced |
| early stopping metric | binary logloss |
| early stopping rounds | 50 |

`n_estimators=1000`は上限であり、validation loglossが50回改善しなければearly stoppingする。

## 6. なぜ2モデルをアンサンブルするのか

jointありモデルは同時極端変動を使い、jointなしモデルが実データらしいと誤認する生成データを強く下げられる場合がある。一方、jointなしモデルは、同時極端変動の日数が少し違うだけで判定が大きく変わる問題を緩和する。

mixed_finalの例は次のとおりである。

| mask | jointあり | jointなし | ensemble |
|---|---:|---:|---:|
| mask3 | 0.8245 | 0.7252 | 0.7749 |
| mask6 | 0.8050 | 0.7382 | 0.7716 |
| mask7 | 0.7468 | 0.5638 | 0.6553 |
| mask4 | 0.6895 | 0.5255 | 0.6075 |
| mask10 | 0.2317 | 0.5435 | 0.3876 |
| mask1 | 0.4384 | 0.3101 | 0.3742 |

mask10はjointなしモデルだけでは0.5を超えるが、jointありモデルの低い確率と平均することで0.5未満になる。両モデルがおおむね同意するmask3・4・6・7は0.5を超えたまま残る。

## 7. 評価指標の読み方

### 7.1 AUC

AUCは、モデルが実データを生成データより上位に並べる能力を表す。正式にはROC曲線の下面積であり、この評価では次の確率として解釈できる。

```text
無作為に選んだ実データ1件のprob_real
>
無作為に選んだ生成データ1件のprob_real
となる確率
```

| AUC | おおよその意味 |
|---:|---|
| 1.0 | すべての実データをすべての生成データより上位に置ける |
| 0.5 | ランダムな順位と同程度 |
| 0.0 | 順位が完全に逆 |

AUCは0.5の判定閾値を使用しない。したがって、AUCが1.0でも実データ確率が0.5未満になることはある。これは「順位は正しいが、確率の尺度または閾値が合っていない」状態である。

### 7.2 Logloss

Loglossは、正解ラベルに対してどれだけ適切な確率を出したかを測る。小さいほど良い。

高い確信度で間違えた予測へ大きな罰を与えるため、順位だけを見るAUCとは異なり、確率の値そのものを評価する。

### 7.3 AccuracyとBalanced accuracy

Accuracyは、0.5閾値で正しく分類した割合である。

```text
Accuracy = 正解した件数 / 全件数
```

このデータは実データと生成データの件数が等しくない。多数派の生成データだけを正しく判定してもAccuracyが高くなる可能性があるため、実データRecallと生成データSpecificityを同じ重さで平均するBalanced accuracyも使用する。

```text
Balanced accuracy
= (実データRecall + 生成データSpecificity) / 2
```

### 7.4 実データRecall

実データRecallは、実データを実データとして検出できた割合である。大きいほど実データの見逃しが少ない。

```text
実データRecall
= 実データ判定できた実データ数 / 全実データ数
```

Recallが0の場合、その評価データ内の実データを0.5閾値では1件も実データ判定できなかったことを意味する。

### 7.5 生成データSpecificity

生成データSpecificityは、生成データを生成データとして正しく排除できた割合である。大きいほど生成データの誤検出が少ない。

```text
生成データSpecificity
= 生成データ判定できた生成データ数 / 全生成データ数
```

## 8. 評価結果

### 8.1 B5

| 指標 | 結果 |
|---|---:|
| AUC | 0.9669 |
| Logloss | 0.3881 |
| Accuracy | 0.9368 |
| Balanced accuracy | 0.9144 |
| 実データRecall | 0.8588 |
| 生成データSpecificity | 0.9700 |

### 8.2 mixed_final

0.5を超えたのは次の4件である。

```text
mask3
mask4
mask6
mask7
```

### 8.3 低重複pseudo_final 7セット

| 指標 | 結果 |
|---|---:|
| 平均AUC | 0.9702 |
| 実データ2件がTop-2 | 6 / 7 |
| 生成データがすべて0.5未満 | 7 / 7 |

pseudo_final_01だけは完全分離できていない。詳細は`evaluation/pseudo_final_low_overlap_v2/README.md`を参照する。

## 9. 推論方法

`lightgbm/`をカレントディレクトリとして実行する。

```bash
.venv/bin/python final_model/ensemble_baseline_no_joint_v1/inference/predict.py \
  --input data/evaluation_sets/final/mixed_final_masked.csv \
  --output-dir /tmp/final_model_prediction
```

出力先には次のファイルが作成される。

- `predictions.csv`：各maskの構成モデル確率、アンサンブル確率、判定
- `summary.json`：予測結果のJSON
- `prob_real_ranking.svg`：実データ確率の順位グラフ
- `component_joint/`：jointありモデル単体の出力
- `component_no_joint/`：jointなしモデル単体の出力

## 10. ディレクトリ

- `models/joint/`：jointありモデル、scaler、特徴量・学習設定
- `models/no_joint/`：jointなしモデル、scaler、特徴量・学習設定
- `inference/`：最終モデル専用の推論入口
- `evaluation/b5/`：B5評価
- `evaluation/mixed_final/`：mixed_final評価
- `evaluation/pseudo_final_low_overlap_v2/`：低重複pseudo 7セット評価
- `documentation/`：補足説明

## 11. 注意点

- mixed_finalの結果を確認したうえで最終モデルを選択しているため、mixed_finalは完全な未知テストではない。
- AUCは順位性能であり、0.5閾値の分類性能とは異なる。
- `prob_real`はモデルの出力確率だが、全時代で完全に較正された確率とは限らない。
- jointあり・なしの両モデルが同じ方向に誤るデータは、単純平均では修正できない。
- モデルはSP500とDGS10の1260日窓を前提とする。
