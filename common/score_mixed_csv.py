#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_features import FEATURE_COLUMNS, features_for_pair  # noqa: E402


def read_mixed(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        headers = list(reader.fieldnames or [])

    masks = sorted(
        {name.removesuffix("_sp500") for name in headers if name.endswith("_sp500")},
        key=lambda name: int(name.removeprefix("mask")),
    )
    samples: list[dict[str, object]] = []
    for mask in masks:
        sp_col = f"{mask}_sp500"
        rate_col = f"{mask}_DGS10"
        if rate_col not in headers:
            continue
        sp = [float(row[sp_col]) for row in rows]
        rate = [float(row[rate_col]) for row in rows]
        samples.append({"mask": mask, "n_rows": len(sp), **features_for_pair(sp, rate)})
    return samples


def save_svg(results: list[dict[str, object]], path: Path) -> None:
    ordered = sorted(results, key=lambda row: float(row["prob_real"]), reverse=True)
    width, height = 1040, 700
    left, top, plot_w, plot_h = 90, 70, 900, 500
    bar_w = plot_w / len(ordered)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="520" y="35" text-anchor="middle" font-family="Arial" font-size="24">LightGBM probability of real data</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for tick in range(6):
        value = tick / 5
        y = top + plot_h * (1 - value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#ddd"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 5:.1f}" text-anchor="end" font-family="Arial" font-size="13">{value:.1f}</text>')
    threshold_y = top + plot_h * 0.5
    parts.append(f'<line x1="{left}" y1="{threshold_y}" x2="{left + plot_w}" y2="{threshold_y}" stroke="#b91c1c" stroke-dasharray="6,5"/>')
    for idx, row in enumerate(ordered):
        value = float(row["prob_real"])
        x = left + idx * bar_w + 5
        h = value * plot_h
        y = top + plot_h - h
        color = "#2563eb" if value >= 0.5 else "#f59e0b"
        parts.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 10:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x + (bar_w - 10) / 2:.1f}" y="{y - 7:.1f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
        parts.append(f'<text x="{x + (bar_w - 10) / 2:.1f}" y="{top + plot_h + 25}" text-anchor="middle" font-family="Arial" font-size="13">{row["mask"]}</text>')
    parts.append('</svg>')
    path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs" / "final_mixed_evaluation")
    args = parser.parse_args()

    with (args.model_dir / "lgb_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (args.model_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)

    samples = read_mixed(args.input)
    x = [[float(row[name]) for name in FEATURE_COLUMNS] for row in samples]
    x_scaled = scaler.transform(x)
    probabilities = model.predict_proba(x_scaled)[:, 1]
    contributions = model.predict(x_scaled, pred_contrib=True)
    results = [
        {
            "mask": row["mask"],
            "n_rows": row["n_rows"],
            "prob_real": float(prob),
            "pred_label": int(prob >= 0.5),
        }
        for row, prob in zip(samples, probabilities)
    ]
    results.sort(key=lambda row: float(row["prob_real"]), reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mask", "n_rows", "prob_real", "pred_label"])
        writer.writeheader()
        writer.writerows(results)
    with (args.output_dir / "features.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mask", "n_rows", *FEATURE_COLUMNS])
        writer.writeheader()
        writer.writerows(samples)
    contribution_rows = []
    for sample, values in zip(samples, contributions):
        contribution_rows.append(
            {
                "mask": sample["mask"],
                **{name: float(value) for name, value in zip(FEATURE_COLUMNS, values[:-1])},
                "expected_value": float(values[-1]),
            }
        )
    with (args.output_dir / "feature_contributions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mask", *FEATURE_COLUMNS, "expected_value"])
        writer.writeheader()
        writer.writerows(contribution_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_svg(results, args.output_dir / "prob_real_ranking.svg")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
