# LightGBM stride comparison

The controlled experiment changes only the real-window stride from `20` to
`126`. Input files, 26 features, split ratios, `StandardScaler`, LightGBM
hyperparameters, seed, and early stopping are unchanged.

## Test result

| experiment | AUC | accuracy | balanced accuracy | logloss | real recall | generated specificity |
|---|---:|---:|---:|---:|---:|---:|
| stride 20 | 0.8021 | 0.5910 | 0.4950 | 0.6758 | 0.0000 | 0.9900 |
| stride 126 | 0.7516 | 0.8874 | 0.4925 | 0.2907 | 0.0000 | 0.9850 |

The higher raw accuracy for stride 126 is caused by the more imbalanced test
set (22 real versus 200 generated). Both models classify every test real window
as generated at the default threshold of 0.5. Balanced accuracy and real recall
therefore show no improvement.
