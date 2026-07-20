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

from build_features import (  # noqa: E402
    JOINT_DIAGNOSTIC_COLUMNS,
    JOINT_DIRECTIONAL_FEATURES,
    JOINT_EQUAL_MEAN_FEATURE,
    feature_columns,
    features_for_pair,
)


def read_mixed(
    path: Path,
    rolling_corr_window: int | None = None,
    joint_feature_mode: str = "single_95",
    dependence_feature_mode: str = "none",
) -> list[dict[str, object]]:
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
        samples.append(
            {
                "mask": mask,
                "n_rows": len(sp),
                **features_for_pair(
                    sp, rate, rolling_corr_window, joint_feature_mode,
                    dependence_feature_mode,
                ),
            }
        )
    return samples


def save_svg(
    results: list[dict[str, object]], path: Path, decision_threshold: float = 0.5
) -> None:
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
    threshold_y = top + plot_h * (1.0 - decision_threshold)
    parts.append(f'<line x1="{left}" y1="{threshold_y}" x2="{left + plot_w}" y2="{threshold_y}" stroke="#b91c1c" stroke-dasharray="6,5"/>')
    for idx, row in enumerate(ordered):
        value = float(row["prob_real"])
        x = left + idx * bar_w + 5
        h = value * plot_h
        y = top + plot_h - h
        color = "#2563eb" if value >= decision_threshold else "#f59e0b"
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

    feature_config_path = args.model_dir / "feature_config.json"
    if feature_config_path.exists():
        feature_config = json.loads(feature_config_path.read_text(encoding="utf-8"))
        rolling_corr_window = feature_config.get("rolling_corr_window")
        joint_feature_mode = feature_config.get("joint_feature_mode", "single_95")
        dependence_feature_mode = feature_config.get(
            "dependence_feature_mode", "none"
        )
        feature_names = list(feature_config["feature_columns"])
        decision_threshold = float(feature_config.get("decision_threshold", 0.5))
    else:
        rolling_corr_window = None
        joint_feature_mode = "single_95"
        dependence_feature_mode = "none"
        feature_names = feature_columns()
        decision_threshold = 0.5
    samples = read_mixed(
        args.input, rolling_corr_window, joint_feature_mode,
        dependence_feature_mode,
    )
    x = [[float(row[name]) for name in feature_names] for row in samples]
    x_scaled = scaler.transform(x)
    probabilities = model.predict_proba(x_scaled)[:, 1]
    contributions = model.predict(x_scaled, pred_contrib=True)
    results = [
        {
            "mask": row["mask"],
            "n_rows": row["n_rows"],
            "prob_real": float(prob),
            "pred_label": int(prob >= decision_threshold),
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
        writer = csv.DictWriter(f, fieldnames=["mask", "n_rows", *feature_names])
        writer.writeheader()
        fields = ["mask", "n_rows", *feature_names]
        writer.writerows({field: row[field] for field in fields} for row in samples)
    if joint_feature_mode == "equal_mean_90_99":
        diagnostic_fields = [
            "mask", "n_rows", *JOINT_DIAGNOSTIC_COLUMNS, JOINT_EQUAL_MEAN_FEATURE
        ]
        with (args.output_dir / "joint_percentile_diagnostics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=diagnostic_fields)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in diagnostic_fields} for row in samples
            )
    elif joint_feature_mode == "directional_95":
        diagnostic_fields = ["mask", "n_rows", *JOINT_DIRECTIONAL_FEATURES]
        with (args.output_dir / "joint_directional_diagnostics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=diagnostic_fields)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in diagnostic_fields} for row in samples
            )
    contribution_rows = []
    for sample, values in zip(samples, contributions):
        contribution_rows.append(
            {
                "mask": sample["mask"],
                **{name: float(value) for name, value in zip(feature_names, values[:-1])},
                "expected_value": float(values[-1]),
            }
        )
    with (args.output_dir / "feature_contributions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["mask", *feature_names, "expected_value"])
        writer.writeheader()
        writer.writerows(contribution_rows)
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    save_svg(
        results, args.output_dir / "prob_real_ranking.svg", decision_threshold
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
