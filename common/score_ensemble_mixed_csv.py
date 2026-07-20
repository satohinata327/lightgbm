#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_mixed_csv import save_svg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    models = config["models"]
    if len(models) < 2:
        raise ValueError("ensemble requires at least two models")
    weights = [float(model["weight"]) for model in models]
    if any(weight < 0.0 for weight in weights):
        raise ValueError("ensemble weights must be non-negative")
    if abs(sum(weights) - 1.0) > 1e-12:
        raise ValueError("ensemble weights must sum to 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    component_predictions: dict[str, dict[str, dict[str, object]]] = {}
    for model in models:
        name = str(model["name"])
        component_dir = args.output_dir / f"component_{name}"
        model_dir = ROOT / model["model_dir"]
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "common/score_mixed_csv.py"),
                "--input",
                str(args.input),
                "--model-dir",
                str(model_dir),
                "--output-dir",
                str(component_dir),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        rows = list(csv.DictReader((component_dir / "predictions.csv").open(encoding="utf-8")))
        component_predictions[name] = {row["mask"]: row for row in rows}

    masks = set.intersection(
        *(set(rows) for rows in component_predictions.values())
    )
    if any(set(rows) != masks for rows in component_predictions.values()):
        raise ValueError("component models returned different mask sets")
    threshold = float(config.get("threshold", 0.5))
    results: list[dict[str, object]] = []
    for mask in masks:
        first = component_predictions[str(models[0]["name"])][mask]
        probability = sum(
            float(model["weight"])
            * float(component_predictions[str(model["name"])][mask]["prob_real"])
            for model in models
        )
        row: dict[str, object] = {
            "mask": mask,
            "n_rows": int(first["n_rows"]),
        }
        for model in models:
            name = str(model["name"])
            row[f"prob_{name}"] = float(component_predictions[name][mask]["prob_real"])
        row["prob_real"] = probability
        row["pred_label"] = int(probability >= threshold)
        results.append(row)
    results.sort(key=lambda row: float(row["prob_real"]), reverse=True)

    fields = [
        "mask",
        "n_rows",
        *(f"prob_{model['name']}" for model in models),
        "prob_real",
        "pred_label",
    ]
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    (args.output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (args.output_dir / "ensemble_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_svg(results, args.output_dir / "prob_real_ranking.svg")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
