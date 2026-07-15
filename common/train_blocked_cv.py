#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import random
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_features import FEATURE_COLUMNS, features_for_pair  # noqa: E402
from prepare_splits import read_generated_wide, read_train_series  # noqa: E402


def packages():
    import lightgbm as lgb
    import numpy as np
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        log_loss,
        roc_auc_score,
    )
    from sklearn.preprocessing import StandardScaler

    return lgb, np, StandardScaler, {
        "auc": roc_auc_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "logloss": log_loss,
        "confusion": confusion_matrix,
    }


def block_bounds(n: int, n_blocks: int) -> list[tuple[int, int]]:
    return [(i * n // n_blocks, (i + 1) * n // n_blocks) for i in range(n_blocks)]


def real_block_rows(
    block_no: int,
    start: int,
    end: int,
    dates: list[str],
    values: list[tuple[str, str]],
    window_length: int,
    stride: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    for window_start in range(start, end - window_length + 1, stride):
        window = values[window_start : window_start + window_length]
        sp = [float(row[0]) for row in window]
        dg = [float(row[1]) for row in window]
        rows.append(
            {
                "sample_id": f"real_b{block_no + 1}_{window_start:05d}",
                "source": "real",
                "label": 1,
                "block": block_no,
                "window_start": window_start,
                "window_end": window_start + window_length - 1,
                **features_for_pair(sp, dg),
            }
        )
    manifest = {
        "block": block_no,
        "row_start": start,
        "row_end": end - 1,
        "start_date": dates[start],
        "end_date": dates[end - 1],
        "n_rows": end - start,
        "n_windows": len(rows),
    }
    return rows, manifest


def generated_block_rows(
    block_no: int,
    sample_indices: list[int],
    generated_rows: list[list[str]],
    source_name: str = "generated",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for sample_idx in sample_indices:
        sp = [float(row[2 * sample_idx]) for row in generated_rows]
        dg = [float(row[2 * sample_idx + 1]) for row in generated_rows]
        rows.append(
            {
                "sample_id": f"{source_name}_{sample_idx + 1:04d}",
                "source": source_name,
                "label": 0,
                "block": block_no,
                "window_start": "",
                "window_end": "",
                **features_for_pair(sp, dg),
            }
        )
    return rows


def write_feature_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "sample_id",
        "source",
        "label",
        "block",
        "window_start",
        "window_end",
        *FEATURE_COLUMNS,
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def xy(rows: list[dict[str, object]], np):
    x = np.asarray([[float(row[name]) for name in FEATURE_COLUMNS] for row in rows], dtype=float)
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)
    return x, y


def metrics_for(y, prob, pred, metric_functions) -> dict[str, object]:
    cm = metric_functions["confusion"](y, pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    return {
        "auc": float(metric_functions["auc"](y, prob)),
        "accuracy": float(metric_functions["accuracy"](y, pred)),
        "balanced_accuracy": float(metric_functions["balanced_accuracy"](y, pred)),
        "logloss": float(metric_functions["logloss"](y, prob)),
        "n": int(len(y)),
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "real_recall": tp / (tp + fn) if tp + fn else 0.0,
        "generated_specificity": tn / (tn + fp) if tn + fp else 0.0,
    }


def write_predictions(path: Path, rows, y, prob, pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "source", "label", "block", "pred_label", "prob_real"])
        for row, yt, yp, pr in zip(rows, y, pred, prob):
            writer.writerow(
                [row["sample_id"], row["source"], int(yt), row["block"], int(yp), f"{float(pr):.10g}"]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = ROOT / config["experiment_dir"]
    data_dir = experiment_dir / "data" / "blocked_features"
    model_dir = experiment_dir / "model"
    results_dir = experiment_dir / "results"
    for path in [data_dir, model_dir, results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    lgb, np, StandardScaler, metric_functions = packages()
    train_csv = (ROOT / config["train_csv"]).resolve()
    dates, real_values = read_train_series(train_csv)

    n_blocks = int(config["blocked_cv"]["n_blocks"])
    final_test_block = int(config["blocked_cv"]["final_test_block"])
    validation_blocks = [int(value) for value in config["blocked_cv"]["validation_blocks"]]
    window_length = int(config["window_length"])
    stride = int(config["stride"])

    real_blocks: list[list[dict[str, object]]] = []
    generated_blocks: list[list[dict[str, object]]] = []
    real_manifest = []
    for block_no, (start, end) in enumerate(block_bounds(len(real_values), n_blocks)):
        rows, manifest = real_block_rows(
            block_no, start, end, dates, real_values, window_length, stride
        )
        real_blocks.append(rows)
        real_manifest.append(manifest)
        write_feature_rows(data_dir / f"real_block_{block_no + 1}.csv", rows)
    generated_manifest: list[dict[str, object]] = []
    if "generated_sources" in config:
        generated_blocks = [[] for _ in range(n_blocks)]
        seed = int(config.get("generated_sampling_seed", 42))
        for source_no, source in enumerate(config["generated_sources"]):
            source_name = str(source["name"])
            source_path = (ROOT / source["wide_csv"]).resolve()
            generated_header, generated_values = read_generated_wide(source_path)
            n_generated = len(generated_header) // 2
            samples_per_block = int(source["samples_per_block"])
            needed = n_blocks * samples_per_block
            if needed > n_generated:
                raise ValueError(
                    f"{source_name} needs {needed} disjoint samples, but only {n_generated} are available"
                )
            indices = list(range(n_generated))
            random.Random(seed + source_no).shuffle(indices)
            for block_no in range(n_blocks):
                selected = indices[
                    block_no * samples_per_block : (block_no + 1) * samples_per_block
                ]
                generated_blocks[block_no].extend(
                    generated_block_rows(
                        block_no, selected, generated_values, source_name=source_name
                    )
                )
                generated_manifest.append(
                    {
                        "block": block_no,
                        "source": source_name,
                        "n_samples": len(selected),
                        "sample_indices_zero_based": selected,
                    }
                )
        for block_no, rows in enumerate(generated_blocks):
            write_feature_rows(data_dir / f"generated_block_{block_no + 1}.csv", rows)
    else:
        generated_csv = (ROOT / config["generated_wide_csv"]).resolve()
        generated_header, generated_values = read_generated_wide(generated_csv)
        n_generated = len(generated_header) // 2
        for block_no, (start, end) in enumerate(block_bounds(n_generated, n_blocks)):
            rows = generated_block_rows(
                block_no, list(range(start, end)), generated_values
            )
            generated_blocks.append(rows)
            generated_manifest.append(
                {"block": block_no, "source": "generated", "n_samples": len(rows)}
            )
            write_feature_rows(data_dir / f"generated_block_{block_no + 1}.csv", rows)

    fold_summaries = []
    best_iterations = []
    model_params = dict(config["lightgbm_params"])
    model_params.setdefault("verbosity", -1)
    stopping = config["early_stopping"]
    for fold_no, validation_block in enumerate(validation_blocks, start=1):
        train_block_ids = list(range(validation_block))
        train_rows = [row for idx in train_block_ids for row in real_blocks[idx] + generated_blocks[idx]]
        val_rows = real_blocks[validation_block] + generated_blocks[validation_block]
        x_train, y_train = xy(train_rows, np)
        x_val, y_val = xy(val_rows, np)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        model = lgb.LGBMClassifier(**model_params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric=stopping["eval_metric"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(stopping["stopping_rounds"]), verbose=False
                )
            ],
        )
        prob = model.predict_proba(x_val)[:, 1]
        pred = model.predict(x_val)
        fold_metrics = metrics_for(y_val, prob, pred, metric_functions)
        best_iteration = int(model.best_iteration_)
        best_iterations.append(best_iteration)
        fold_summary = {
            "fold": fold_no,
            "train_blocks": train_block_ids,
            "validation_block": validation_block,
            "train_real": sum(len(real_blocks[idx]) for idx in train_block_ids),
            "train_generated": sum(len(generated_blocks[idx]) for idx in train_block_ids),
            "validation_real": len(real_blocks[validation_block]),
            "validation_generated": len(generated_blocks[validation_block]),
            "best_iteration": best_iteration,
            "metrics": fold_metrics,
        }
        fold_summaries.append(fold_summary)
        fold_dir = results_dir / f"fold_{fold_no}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        (fold_dir / "metrics.json").write_text(json.dumps(fold_summary, indent=2), encoding="utf-8")
        write_predictions(fold_dir / "validation_predictions.csv", val_rows, y_val, prob, pred)

    final_n_estimators = max(1, int(statistics.median(best_iterations)))
    final_train_blocks = list(range(final_test_block))
    final_train_rows = [
        row for idx in final_train_blocks for row in real_blocks[idx] + generated_blocks[idx]
    ]
    final_test_rows = real_blocks[final_test_block] + generated_blocks[final_test_block]
    x_train, y_train = xy(final_train_rows, np)
    x_test, y_test = xy(final_test_rows, np)
    final_scaler = StandardScaler()
    x_train = final_scaler.fit_transform(x_train)
    x_test = final_scaler.transform(x_test)
    final_params = dict(model_params)
    final_params["n_estimators"] = final_n_estimators
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(x_train, y_train)
    test_prob = final_model.predict_proba(x_test)[:, 1]
    test_pred = final_model.predict(x_test)
    test_metrics = metrics_for(y_test, test_prob, test_pred, metric_functions)

    with (model_dir / "lgb_model.pkl").open("wb") as f:
        pickle.dump(final_model, f)
    with (model_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(final_scaler, f)
    write_predictions(results_dir / "test_predictions.csv", final_test_rows, y_test, test_prob, test_pred)

    cv_auc = [float(fold["metrics"]["auc"]) for fold in fold_summaries]
    cv_recall = [float(fold["metrics"]["real_recall"]) for fold in fold_summaries]
    summary = {
        "experiment_name": config["experiment_name"],
        "real_blocks": real_manifest,
        "generated_blocks": [
            {"block": idx, "n_samples": len(rows)} for idx, rows in enumerate(generated_blocks)
        ],
        "generated_source_assignments": generated_manifest,
        "folds": fold_summaries,
        "cv": {
            "auc_mean": statistics.mean(cv_auc),
            "auc_std": statistics.stdev(cv_auc) if len(cv_auc) > 1 else 0.0,
            "auc_min": min(cv_auc),
            "real_recall_mean": statistics.mean(cv_recall),
            "best_iterations": best_iterations,
            "final_n_estimators": final_n_estimators,
        },
        "final": {
            "train_blocks": final_train_blocks,
            "test_block": final_test_block,
            "train_real": sum(len(real_blocks[idx]) for idx in final_train_blocks),
            "train_generated": sum(len(generated_blocks[idx]) for idx in final_train_blocks),
            "test_real": len(real_blocks[final_test_block]),
            "test_generated": len(generated_blocks[final_test_block]),
            "metrics": test_metrics,
        },
        "effective_lightgbm_params": final_params,
    }
    (results_dir / "blocked_cv_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (results_dir / "metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
