# ZA＋MF2 Blocked CV LightGBMモデル解説

## 1. モデルの概要

このモデルは、SP500と米国10年国債金利（DGS10）の1260日分の時系列が、実データか生成データかを判別する二値分類モデルである。

- 実データ：`train_data/train_sp500_us10y.csv`
- 生成データ：`za_final7`と`mf2_garch_regime_corr`
- 正例（label=1）：実データ
- 負例（label=0）：生成データ
- 入力：1260日分のSP500・DGS10から計算した26特徴量
- 分類器：LightGBM
- 実データの窓幅：`window_length=1260`
- 実データの窓移動幅：`stride=20`
- 判定閾値：実データ確率0.5

このモデルの重要な変更点は、負例をZAだけに限定せず、ZAとMF2を同数使用したことである。特定の生成方法だけに存在する癖を学習するのではなく、複数の生成モデルに共通する「実データとの違い」を学習させることを狙っている。

## 2. 学習データの構成

### 2.1 実データ

実データ全体を時系列順に5つの非重複blockへ分割している。blockをまたいで同じ期間が現れないため、近接した窓が学習データとvalidationに混入する通常のランダム分割よりも厳しい評価になる。

| Block | 期間 | 元データ行数 | 1260日窓数 | 用途 |
|---|---|---:|---:|---|
| B1 | 1966-01-03〜1978-01-30 | 2946 | 85 | 学習開始block |
| B2 | 1978-01-31〜1990-01-29 | 2947 | 85 | Validation fold 1 |
| B3 | 1990-01-30〜2001-12-26 | 2947 | 85 | Validation fold 2 |
| B4 | 2001-12-27〜2013-11-14 | 2947 | 85 | Validation fold 3 |
| B5 | 2013-11-15〜2025-10-02 | 2947 | 85 | 最終test専用 |

各blockの中だけで1260日窓を作るため、異なるblock間で実データ期間は重複しない。`stride=20`は、同一block内では窓の開始位置を20営業日ずつ移動することを意味する。

### 2.2 生成データ

各blockへ次の200系列を割り当てている。

- ZA：100系列
- MF2：100系列

両ソースとも1000系列からseed 42で決定的にシャッフルし、各blockへ100系列ずつ割り当てた。合計でZA 500系列、MF2 500系列を使用している。同じ生成系列を複数blockへ再利用していないため、trainで見た生成系列がvalidationやB5 testへ漏れることはない。

生成データ総数は従来のZAのみモデルと同じ1 blockあたり200系列に維持している。したがって、単純に負例数を増やしたモデルではなく、負例200系列の内訳を多様化したモデルである。ただしZA系列の抽出方法も従来の連番分割からseed固定のランダム抽出へ変わっているため、厳密には生成ソースの変更だけでなく、ZAサンプル構成の差も含まれる。

## 3. 時代block型cross-validation

過去のblockだけで学習し、その直後の未知時代をvalidationにする拡張窓方式を採用している。

| Fold | 学習block | Validation block |
|---|---|---|
| 1 | B1 | B2 |
| 2 | B1＋B2 | B3 |
| 3 | B1＋B2＋B3 | B4 |

B5はハイパーパラメータ選択、early stopping、木の本数の決定には使用せず、最後の一度だけtestとして評価する。`mixed_final_masked.csv`も学習、early stopping、モデル選択には一切使用していない。

## 4. 26特徴量

LightGBMへ1260×2の生時系列を直接入力するのではなく、各系列ペアを次の26個の統計量へ変換している。

### 4.1 SP500の分布特徴（7個）

| 特徴量 | 内容 |
|---|---|
| `sp_mean` | 平均 |
| `sp_std` | 標本標準偏差 |
| `sp_skew` | 歪度 |
| `sp_kurt` | 超過尖度 |
| `sp_q90_abs` | 絶対値の90%分位点 |
| `sp_q95_abs` | 絶対値の95%分位点 |
| `sp_q99_abs` | 絶対値の99%分位点 |

### 4.2 DGS10の分布特徴（7個）

`rate_mean`、`rate_std`、`rate_skew`、`rate_kurt`、`rate_q90_abs`、`rate_q95_abs`、`rate_q99_abs`を計算する。意味はSP500側と同じである。

### 4.3 時系列依存特徴（8個）

- SP500：lag 1、lag 5の自己相関
- SP500絶対値：lag 1、lag 5の自己相関
- DGS10：lag 1、lag 5の自己相関
- DGS10絶対値：lag 1、lag 5の自己相関

元系列の自己相関は値の方向の連続性を、絶対値自己相関はボラティリティ・クラスタリングを簡易的に表現する。

### 4.4 2変量・裾特徴（4個）

| 特徴量 | 内容 |
|---|---|
| `corr_sp_rate` | SP500とDGS10の同時点相関 |
| `joint_large_move_95` | 両系列の絶対値がそれぞれ95%分位点を同時に超える割合 |
| `sp_tail_ratio` | SP500の絶対値99%分位点÷標準偏差 |
| `rate_tail_ratio` | DGS10の絶対値99%分位点÷標準偏差 |

すべての特徴量は、各foldの学習データだけで適合した`StandardScaler`により標準化する。Validationとtestには学習側で得た平均・標準偏差を適用する。

## 5. LightGBMの学習設定

| パラメータ | 値 | 役割 |
|---|---:|---|
| `objective` | `binary` | 二値分類 |
| `learning_rate` | 0.03 | 1本の木が加える更新量 |
| `max_depth` | 4 | 木の最大深さ |
| `num_leaves` | 15 | 1本の木の最大葉数 |
| `min_child_samples` | 20 | 葉に必要な最小サンプル数 |
| `n_estimators` | 1000 | early stopping前の最大木数 |
| `colsample_bytree` | 0.8 | 木ごとに使用する特徴量割合 |
| `subsample` | 0.8 | baggingで使用するサンプル割合 |
| `subsample_freq` | 1 | 毎iterationでbagging |
| `reg_alpha` | 0.0 | L1正則化なし |
| `reg_lambda` | 0.0 | L2正則化なし |
| `class_weight` | `balanced` | クラス数に応じて重みを補正 |
| `random_state` | 42 | 乱数を固定 |

目的関数はbinary loglossである。実データである確率を $p_i$、正解ラベルを $y_i \in \{0,1\}$ とすると、基本形は次の通りである。

$$
-\frac{1}{N}\sum_i\left[y_i\log p_i+(1-y_i)\log(1-p_i)\right]
$$

実際には`class_weight=balanced`が指定されているため、サンプル数の少ない実データ側へ大きい重みが付く。評価時の`prob_real`はLightGBMが出力したlabel 1、すなわち実データ側の確率である。

各foldではvalidation binary loglossが50 iteration改善しなければearly stoppingする。得られたbest iterationは17、49、36であり、その中央値36本を最終モデルの木の本数に採用した。最終モデルはB1〜B4をすべて使い、36本の木で再学習している。

## 6. 評価結果

### 6.1 時代別validation

| Fold | AUC | 正解率 | Balanced accuracy | Logloss | 実データRecall | 生成データSpecificity |
|---|---:|---:|---:|---:|---:|---:|
| B2 | 0.5426 | 0.6982 | 0.4975 | 0.6110 | 0.0000 | 0.9950 |
| B3 | 0.7915 | 0.9018 | 0.8387 | 0.3614 | 0.6824 | 0.9950 |
| B4 | 0.9631 | 0.6877 | 0.4900 | 0.5086 | 0.0000 | 0.9800 |

- AUC平均：0.7657
- AUC標準偏差：0.2114
- 最低AUC：0.5426

AUCは確率の順位付け性能を表す。B4ではAUC 0.9631と高い一方、0.5閾値では実データを1件も実データと判定できていない。これは順位付け自体は良くても、確率の水準または閾値が時代によってずれていることを示す。したがって、AUCが高いことと0.5閾値で正しく分類できることは別に評価する必要がある。

### 6.2 最終B5 test

| 指標 | 値 |
|---|---:|
| AUC | 0.9654 |
| 正解率 | 0.9298 |
| Balanced accuracy | 0.9060 |
| Logloss | 0.3354 |
| 実データRecall | 0.8471 |
| 生成データSpecificity | 0.9650 |

混同行列は次の通りである。

|  | 実データと予測 | 生成データと予測 |
|---|---:|---:|
| 実データ85件 | 72 | 13 |
| 生成データ200件 | 7 | 193 |

生成ソース別では、ZAを93/100件、MF2を100/100件生成データと判定した。MF2に対する識別は非常に強い一方、B5の実データ13窓を生成データと誤判定している。

### 6.3 ZAのみモデルとの比較

| 指標 | ZAのみ | ZA＋MF2 |
|---|---:|---:|
| CV平均AUC | 0.6628 | 0.7657 |
| CV最低AUC | 0.4661 | 0.5426 |
| CV AUC標準偏差 | 0.2026 | 0.2114 |
| B5 AUC | 0.9906 | 0.9654 |
| B5 Balanced accuracy | 0.9825 | 0.9060 |
| B5実データRecall | 1.0000 | 0.8471 |

複数生成モデルを学習させたことでCVの平均・最低AUCは改善したが、B5の分類性能は悪化した。一方、後述するmixed finalでは候補maskの分離が大きく改善している。つまり、すべての評価指標で一律に優れたモデルではなく、未知生成データへの分離という目的に対して現時点で最も有望なモデルと位置付けるのが正確である。

## 7. mixed finalの判定結果

| 順位 | Mask | 実データ確率 | 0.5閾値での判定 |
|---:|---|---:|---|
| 1 | mask3 | 0.8245 | 実データ |
| 2 | mask6 | 0.8050 | 実データ |
| 3 | mask7 | 0.7468 | 実データ |
| 4 | mask4 | 0.6895 | 実データ |
| 5 | mask1 | 0.4384 | 生成データ |
| 6 | mask9 | 0.2868 | 生成データ |
| 7 | mask12 | 0.2788 | 生成データ |
| 8 | mask14 | 0.2428 | 生成データ |
| 9 | mask10 | 0.2317 | 生成データ |
| 10 | mask2 | 0.2253 | 生成データ |
| 11 | mask8 | 0.1742 | 生成データ |
| 12 | mask13 | 0.1710 | 生成データ |
| 13 | mask5 | 0.1664 | 生成データ |
| 14 | mask11 | 0.1664 | 生成データ |

mask3、mask4、mask6、mask7だけが0.5を超え、上位4つを占めた。ZAのみモデルではmask1、mask2、mask8、mask10なども0.5を超えていたため、負例の多様化によってmixed final上の分離は明確になった。

ただし、これらの値を統計的に校正済みの確率とみなすことはできない。LightGBMの`predict_proba`は順位付けには利用できるが、blocked CVで時代ごとの確率水準が変動しているため、「0.82なら82%の確率で実データ」とそのまま解釈すべきではない。

## 8. このモデルの強み

1. 実データを時系列blockで分け、trainとvalidationの期間重複を防いでいる。
2. ZAとMF2を同数使用し、単一生成モデルへの過度な特化を緩和している。
3. 生成系列もblock間で重複せず、生成データ側のリークを防いでいる。
4. B5とmixed finalをモデル選択から分離している。
5. mixed final上でmask3、4、6、7とその他のmaskを初めて0.5境界で明確に分離した。

## 9. 現時点の限界

1. 時代別AUCの標準偏差が0.2114あり、時代依存性がまだ大きい。
2. B2とB4では、0.5閾値で実データRecallが0になっている。
3. B5では13/85の実データ窓を生成データと誤判定する。
4. 特徴量は1260日全体の集約統計量であり、期間内の局面変化や特徴の並び順を十分には表現できない。
5. 負例はZAとMF2だけであり、TimeGAN、APARCH、GC-GARCHなど未知の生成原理への汎化は未確認である。
6. 確率校正を行っていないため、`prob_real`の絶対値より順位と相対差を重視すべきである。
7. mixed finalの正解ラベルは学習時に使用していないが、今後この結果を見ながらモデルを繰り返し選ぶと、間接的にmixed finalへ過適合する危険がある。

## 10. 関連ファイル

- 学習設定：`lightgbm/train/stride20_blocked_cv_za_mf2/config.json`
- 学習実装：`lightgbm/common/train_blocked_cv.py`
- 特徴量実装：`lightgbm/common/build_features.py`
- CV・B5詳細：`lightgbm/train/stride20_blocked_cv_za_mf2/results/blocked_cv_summary.json`
- B5予測：`lightgbm/train/stride20_blocked_cv_za_mf2/results/test_predictions.csv`
- 生成ソース別評価：`lightgbm/train/stride20_blocked_cv_za_mf2/results/source_metrics.json`
- mixed final予測：`lightgbm/test/stride20_blocked_cv_za_mf2__mixed_final/predictions.csv`
- mixed final特徴量：`lightgbm/test/stride20_blocked_cv_za_mf2__mixed_final/features.csv`
- mixed finalランキング図：`lightgbm/test/stride20_blocked_cv_za_mf2__mixed_final/prob_real_ranking.svg`

## 11. 再実行方法

学習：

```bash
LOKY_MAX_CPU_COUNT=8 lightgbm/.venv/bin/python \
  lightgbm/common/train_blocked_cv.py \
  --config lightgbm/train/stride20_blocked_cv_za_mf2/config.json
```

mixed final評価：

```bash
LOKY_MAX_CPU_COUNT=8 lightgbm/.venv/bin/python \
  lightgbm/common/score_mixed_csv.py \
  --input mixed_final_masked.csv \
  --model-dir lightgbm/train/stride20_blocked_cv_za_mf2/model \
  --output-dir lightgbm/test/stride20_blocked_cv_za_mf2__mixed_final
```
