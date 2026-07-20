#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from prepare_splits import read_generated_wide, read_train_series  # noqa: E402


def used_indices(summary: dict[str, object], source: str) -> set[int]:
    return {
        int(index)
        for assignment in summary["generated_source_assignments"]
        if assignment["source"] == source
        for index in assignment["sample_indices_zero_based"]
    }


def write_mixed(path: Path, samples: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        field
        for mask_no in range(1, len(samples) + 1)
        for field in (f"mask{mask_no}_sp500", f"mask{mask_no}_DGS10")
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row_no in range(1260):
            row: dict[str, str] = {}
            for mask_no, sample in enumerate(samples, start=1):
                row[f"mask{mask_no}_sp500"] = sample["sp"][row_no]
                row[f"mask{mask_no}_DGS10"] = sample["rate"][row_no]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data" / "evaluation_sets" / "pseudo_final_za_mf2_unused",
    )
    parser.add_argument("--n-sets", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--exclude-suite-dir", type=Path)
    args = parser.parse_args()

    baseline_summary_path = (
        ROOT
        / "train"
        / "stride20_blocked_cv_za_mf2"
        / "results"
        / "blocked_cv_summary.json"
    )
    baseline_summary = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
    sources = {
        "za_final7": WORKSPACE / "za_final7_1000paths_wide.csv",
        "mf2_garch_regime_corr": (
            WORKSPACE
            / "garch_origin/models/mf2_garch_regime_corr/runs/run_001"
            / "generated_1000_wide/mf2_garch_regime_corr_1000x1260_wide.csv"
        ),
    }
    generated: dict[str, list[list[str]]] = {}
    available: dict[str, list[int]] = {}
    rng = random.Random(args.seed)
    for source, path in sources.items():
        header, rows = read_generated_wide(path)
        generated[source] = rows
        excluded = used_indices(baseline_summary, source)
        if args.exclude_suite_dir:
            for manifest_path in args.exclude_suite_dir.glob("pseudo_final_*_manifest.csv"):
                with manifest_path.open(newline="", encoding="utf-8") as f:
                    for row in csv.DictReader(f):
                        if row["source"] == source:
                            excluded.add(int(row["source_index_zero_based"]))
        unused = sorted(set(range(len(header) // 2)) - excluded)
        rng.shuffle(unused)
        if len(unused) < args.n_sets * 6:
            raise ValueError(f"{source}: unused paths are insufficient")
        available[source] = unused

    dates, real_values = read_train_series(WORKSPACE / "train_data/train_sp500_us10y.csv")
    b5_start = 4 * len(real_values) // 5
    b5_end = len(real_values)
    last_start = b5_end - 1260
    n_real_windows = args.n_sets * 2
    if n_real_windows < 2:
        raise ValueError("at least two real windows are required")
    b5_window_starts = [
        round(b5_start + index * (last_start - b5_start) / (n_real_windows - 1))
        for index in range(n_real_windows)
    ]
    minimum_start_gap = min(
        right - left for left, right in zip(b5_window_starts, b5_window_starts[1:])
    )
    maximum_overlap_fraction = max(0, 1260 - minimum_start_gap) / 1260
    real_positions = list(range(args.n_sets)) + [
        n_real_windows - 1 - index for index in range(args.n_sets)
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suite_manifest: dict[str, object] = {
        "description": "B5 real 2 + unused ZA 6 + unused MF2 6 per set",
        "seed": args.seed,
        "n_sets": args.n_sets,
        "baseline_summary": str(baseline_summary_path.relative_to(ROOT)),
        "generated_paths_are_unique_across_sets": True,
        "excluded_previous_suite": (
            str(args.exclude_suite_dir) if args.exclude_suite_dir else None
        ),
        "minimum_real_window_start_gap_rows": minimum_start_gap,
        "maximum_adjacent_real_window_overlap_fraction": maximum_overlap_fraction,
        "real_window_note": (
            "All real windows are held-out B5 windows. Because each window is 1260 rows "
            "and B5 has 2947 rows, real windows overlap across sets and are not independent."
        ),
        "sets": [],
    }
    for set_index in range(args.n_sets):
        samples: list[dict[str, object]] = []
        for real_no in range(2):
            position = real_positions[set_index + real_no * args.n_sets]
            start = b5_window_starts[position]
            window = real_values[start : start + 1260]
            samples.append(
                {
                    "kind": "real",
                    "source": "real_b5",
                    "source_index": "",
                    "window_start": start,
                    "window_end": start + 1259,
                    "start_date": dates[start],
                    "end_date": dates[start + 1259],
                    "sp": [row[0] for row in window],
                    "rate": [row[1] for row in window],
                }
            )
        for source in sources:
            for offset in range(6):
                sample_index = available[source][set_index * 6 + offset]
                rows = generated[source]
                samples.append(
                    {
                        "kind": "generated",
                        "source": source,
                        "source_index": sample_index,
                        "window_start": "",
                        "window_end": "",
                        "start_date": "",
                        "end_date": "",
                        "sp": [row[2 * sample_index] for row in rows],
                        "rate": [row[2 * sample_index + 1] for row in rows],
                    }
                )
        rng.shuffle(samples)
        set_name = f"pseudo_final_{set_index + 1:02d}"
        csv_path = args.output_dir / f"{set_name}.csv"
        write_mixed(csv_path, samples)
        manifest_rows = []
        for mask_no, sample in enumerate(samples, start=1):
            manifest_rows.append(
                {
                    "mask": f"mask{mask_no}",
                    "label": int(sample["kind"] == "real"),
                    "kind": sample["kind"],
                    "source": sample["source"],
                    "source_index_zero_based": sample["source_index"],
                    "window_start": sample["window_start"],
                    "window_end": sample["window_end"],
                    "start_date": sample["start_date"],
                    "end_date": sample["end_date"],
                }
            )
        manifest_path = args.output_dir / f"{set_name}_manifest.csv"
        with manifest_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
            writer.writeheader()
            writer.writerows(manifest_rows)
        suite_manifest["sets"].append(
            {
                "name": set_name,
                "csv": csv_path.name,
                "manifest": manifest_path.name,
                "real_masks": [row["mask"] for row in manifest_rows if row["label"] == 1],
            }
        )

    (args.output_dir / "suite_manifest.json").write_text(
        json.dumps(suite_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    final_dir = ROOT / "data" / "evaluation_sets" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(WORKSPACE / "mixed_final_masked.csv", final_dir / "mixed_final_masked.csv")
    print(json.dumps(suite_manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
