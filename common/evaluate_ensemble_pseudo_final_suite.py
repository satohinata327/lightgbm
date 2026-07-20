#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-dir", type=Path, required=True)
    parser.add_argument("--ensemble-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from sklearn.metrics import roc_auc_score

    suite = json.loads((args.suite_dir / "suite_manifest.json").read_text(encoding="utf-8"))
    summaries = []
    for set_info in suite["sets"]:
        name = set_info["name"]
        output = args.output_dir / name
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "common/score_ensemble_mixed_csv.py"),
                "--config",
                str(args.ensemble_config),
                "--input",
                str(args.suite_dir / set_info["csv"]),
                "--output-dir",
                str(output),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        predictions = json.loads((output / "summary.json").read_text(encoding="utf-8"))
        labels = {
            row["mask"]: int(row["label"])
            for row in csv.DictReader(
                (args.suite_dir / set_info["manifest"]).open(encoding="utf-8")
            )
        }
        ordered = sorted(predictions, key=lambda row: float(row["prob_real"]), reverse=True)
        real_probs = [float(row["prob_real"]) for row in ordered if labels[row["mask"]] == 1]
        generated_probs = [float(row["prob_real"]) for row in ordered if labels[row["mask"]] == 0]
        y = [labels[row["mask"]] for row in ordered]
        prob = [float(row["prob_real"]) for row in ordered]
        summaries.append(
            {
                "set": name,
                "auc": float(roc_auc_score(y, prob)),
                "top2_real_count": sum(labels[row["mask"]] for row in ordered[:2]),
                "top4_real_count": sum(labels[row["mask"]] for row in ordered[:4]),
                "real_min_prob": min(real_probs),
                "generated_max_prob": max(generated_probs),
                "separation_margin": min(real_probs) - max(generated_probs),
                "real_above_0_5": sum(value >= 0.5 for value in real_probs),
                "generated_above_0_5": sum(value >= 0.5 for value in generated_probs),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "suite_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    aggregate = {
        "ensemble_config": str(args.ensemble_config),
        "n_sets": len(summaries),
        "mean_auc": sum(row["auc"] for row in summaries) / len(summaries),
        "perfect_top2_sets": sum(row["top2_real_count"] == 2 for row in summaries),
        "all_real_in_top4_sets": sum(row["top4_real_count"] == 2 for row in summaries),
        "positive_margin_sets": sum(row["separation_margin"] > 0 for row in summaries),
        "mean_margin": sum(row["separation_margin"] for row in summaries) / len(summaries),
        "all_real_above_0_5_sets": sum(row["real_above_0_5"] == 2 for row in summaries),
        "no_generated_above_0_5_sets": sum(
            row["generated_above_0_5"] == 0 for row in summaries
        ),
        "sets": summaries,
    }
    (args.output_dir / "suite_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
