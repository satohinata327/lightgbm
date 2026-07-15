# LightGBM real/generated detection

This workspace separates model training from evaluation so every result path
identifies both the model and target dataset.

## Layout

```text
common/       shared feature, training, scoring, and plotting code
train/        training configurations, processed data, models, and train metrics
test/         evaluation results named <model>__<target>
comparison/   comparisons across models or evaluations
```

## Train a model

Baseline stride 20:

```bash
.venv/bin/python common/prepare_splits.py --config train/baseline_stride20/config.json
.venv/bin/python common/build_features.py --config train/baseline_stride20/config.json
.venv/bin/python common/train_classifier.py --config train/baseline_stride20/config.json
.venv/bin/python common/plot_score_distribution.py --config train/baseline_stride20/config.json
```

For stride 126, replace the config path with
`train/stride126/config.json`. The two configurations differ in stride only.

## Evaluate a target

```bash
.venv/bin/python common/score_mixed_csv.py \
  --input ../mixed_final_masked.csv \
  --model-dir train/baseline_stride20/model \
  --output-dir test/baseline_stride20__mixed_final
```

Evaluation directory names follow `<model_name>__<target_data_name>`.

## Data policy

Raw sources remain in the DSS workspace. Experiment-specific splits and
features currently stay under `train/<model>/data/`. This can later be replaced
by a shared raw-data catalog without changing test result naming.
