# LightGBM 実データ・生成データ判別

このワークスペースではモデル学習と評価を分離し、各結果パスから使用モデルと評価対象データの両方が分かる構成にしている。

## ディレクトリ構成

```text
common/       特徴量作成・学習・スコア計算・描画の共通コード
train/        学習設定、加工済みデータ、モデル、学習評価結果
test/         <model>__<target>形式で命名した評価結果
comparison/   モデル間または評価間の比較
```

## モデルの学習

ベースライン（stride 20）：

```bash
.venv/bin/python common/prepare_splits.py --config train/baseline_stride20/config.json
.venv/bin/python common/build_features.py --config train/baseline_stride20/config.json
.venv/bin/python common/train_classifier.py --config train/baseline_stride20/config.json
.venv/bin/python common/plot_score_distribution.py --config train/baseline_stride20/config.json
```

stride 126を使用する場合は、設定パスを`train/stride126/config.json`へ置き換える。この2つの設定の違いはstrideだけである。

`train/stride20_nonoverlap_mf2/config.json`ではstride 20を維持しつつ、`period_then_window`方式を使用する。先に元データを時系列期間で分割し、それぞれの期間内だけで窓を作るため、異なる分割間で実データ窓が重複しない。この実験では負例にMF2だけを使用し、複数の生成モデルを追加する前に分割方法の効果を測定する。

`train/stride20_nonoverlap_za_final7/config.json`では、非重複の実データ期間とモデル設定を維持し、負例だけを`za_final7_1000paths_wide.csv`へ置き換える。

`train/stride20_blocked_cv_za_final7/config.json`では、5つの非重複時代block、3つの拡張型validation fold、最後まで未使用の2013〜2025年test blockを使用する。実行には`common/train_blocked_cv.py`を使用する。

## 評価対象データの判定

```bash
.venv/bin/python common/score_mixed_csv.py \
  --input ../mixed_final_masked.csv \
  --model-dir train/baseline_stride20/model \
  --output-dir test/baseline_stride20__mixed_final
```

評価ディレクトリ名は`<model_name>__<target_data_name>`形式とする。

## データ配置方針

元データはDSSワークスペースに残す。現在、実験固有の分割データと特徴量は`train/<model>/data/`配下に保存している。将来、test結果の命名規則を変えずに、共有の元データカタログへ置き換えることもできる。
