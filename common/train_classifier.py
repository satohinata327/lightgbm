#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = ROOT / "data" / "features" / "mf2_garch_regime_corr_baseline"
RUN_DIR = ROOT / "runs" / "mf2_garch_regime_corr_baseline"


def require_packages():
    try:
        import lightgbm as lgb
        import numpy as np
        from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install lightgbm scikit-learn numpy\n"
            f"Original error: {exc}"
        ) from exc
    return lgb, np, StandardScaler, roc_auc_score, accuracy_score, log_loss


def read_feature_csv(path: Path, feature_cols: list[str]):
    rows: list[dict[str, str]] = []
    x: list[list[float]] = []
    y: list[int] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            x.append([float(row[col]) for col in feature_cols])
            y.append(int(row["label"]))
    return rows, x, y


def save_predictions(path: Path, rows: list[dict[str, str]], y_true, y_pred, y_prob) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "split", "source", "label", "pred_label", "prob_real"])
        for row, yt, yp, prob in zip(rows, y_true, y_pred, y_prob):
            writer.writerow([row["sample_id"], row["split"], row["source"], yt, int(yp), f"{float(prob):.10g}"])


def main() -> None:
    global FEATURE_DIR, RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = ROOT / config["experiment_dir"]
    FEATURE_DIR = experiment_dir / "data" / "features"
    RUN_DIR = experiment_dir
    lgb, np, StandardScaler, roc_auc_score, accuracy_score, log_loss = require_packages()
    manifest = json.loads((FEATURE_DIR / "feature_manifest.json").read_text(encoding="utf-8"))
    feature_cols = list(manifest["features"])
    train_rows, x_train, y_train = read_feature_csv(FEATURE_DIR / "train_features.csv", feature_cols)
    val_rows, x_val, y_val = read_feature_csv(FEATURE_DIR / "val_features.csv", feature_cols)
    test_rows, x_test, y_test = read_feature_csv(FEATURE_DIR / "test_features.csv", feature_cols)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(np.asarray(x_train, dtype=float))
    x_val_scaled = scaler.transform(np.asarray(x_val, dtype=float))
    x_test_scaled = scaler.transform(np.asarray(x_test, dtype=float))

    model = lgb.LGBMClassifier(
        objective="binary",
        random_state=42,
        class_weight="balanced",
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=20,
        n_estimators=1000,
        colsample_bytree=0.8,
        subsample=0.8,
        subsample_freq=1,
        verbosity=-1,
    )
    model.fit(
        x_train_scaled,
        np.asarray(y_train),
        eval_set=[(x_val_scaled, np.asarray(y_val))],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True)],
    )

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = RUN_DIR / "model"
    results_dir = RUN_DIR / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "lgb_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (model_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)

    metrics = {}
    for split, rows, x_scaled, y in [
        ("train", train_rows, x_train_scaled, y_train),
        ("val", val_rows, x_val_scaled, y_val),
        ("test", test_rows, x_test_scaled, y_test),
    ]:
        prob = model.predict_proba(x_scaled)[:, 1]
        pred = model.predict(x_scaled)
        metrics[split] = {
            "auc": float(roc_auc_score(y, prob)),
            "accuracy": float(accuracy_score(y, pred)),
            "logloss": float(log_loss(y, prob)),
            "n": len(y),
        }
        save_predictions(results_dir / f"{split}_predictions.csv", rows, y, pred, prob)

    importances = sorted(
        zip(feature_cols, model.feature_importances_),
        key=lambda item: item[1],
        reverse=True,
    )
    with (results_dir / "feature_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(importances)

    (results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (results_dir / "README.md").write_text(
        "# MF2 Regime-Correlation Baseline LightGBM\n\n"
        "Features match `detection_light-gbm.ipynb` baseline feature set.\n\n"
        "## Metrics\n\n"
        + "\n".join(
            f"- {split}: AUC={m['auc']:.4f}, accuracy={m['accuracy']:.4f}, logloss={m['logloss']:.4f}, n={m['n']}"
            for split, m in metrics.items()
        )
        + "\n\n## Top Features\n\n"
        + "\n".join(f"- {name}: {importance}" for name, importance in importances[:15])
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2))
    print("Top features:")
    for name, importance in importances[:15]:
        print(name, importance)


if __name__ == "__main__":
    main()
