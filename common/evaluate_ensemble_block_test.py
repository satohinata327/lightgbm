#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_predictions(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return {row["sample_id"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, log_loss, recall_score,
        roc_auc_score,
    )

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["models"]
    weights = [float(model["weight"]) for model in models]
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("ensemble weights must sum to 1")
    components = {
        model["name"]: read_predictions(
            Path(model["experiment_dir"]) / "results" / "test_predictions.csv"
        )
        for model in models
    }
    ids = set.intersection(*(set(rows) for rows in components.values()))
    if any(set(rows) != ids for rows in components.values()):
        raise ValueError("component test predictions have different sample IDs")

    threshold = float(config.get("threshold", 0.5))
    output = []
    for sample_id in sorted(ids):
        first = components[models[0]["name"]][sample_id]
        label = int(first["label"])
        if any(int(components[model["name"]][sample_id]["label"]) != label for model in models):
            raise ValueError(f"label mismatch: {sample_id}")
        probability = sum(
            float(model["weight"])
            * float(components[model["name"]][sample_id]["prob_real"])
            for model in models
        )
        row = {
            "sample_id": sample_id,
            "source": first["source"],
            "label": label,
            "block": first["block"],
        }
        for model in models:
            name = model["name"]
            row[f"prob_{name}"] = float(components[name][sample_id]["prob_real"])
        row["prob_real"] = probability
        row["pred_label"] = int(probability >= threshold)
        output.append(row)

    y = [row["label"] for row in output]
    probability = [row["prob_real"] for row in output]
    prediction = [row["pred_label"] for row in output]
    specificity = sum(a == 0 and b == 0 for a, b in zip(y, prediction)) / sum(a == 0 for a in y)
    summary = {
        "ensemble_config": str(args.config),
        "threshold": threshold,
        "n": len(output),
        "auc": float(roc_auc_score(y, probability)),
        "logloss": float(log_loss(y, probability)),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "real_recall": float(recall_score(y, prediction)),
        "generated_specificity": float(specificity),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fields = list(output[0])
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
