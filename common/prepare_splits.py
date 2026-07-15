#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent

TRAIN_CSV = WORKSPACE / "train_data" / "train_sp500_us10y.csv"
GENERATED_WIDE_CSV = (
    WORKSPACE
    / "garch_origin"
    / "models"
    / "mf2_garch_regime_corr"
    / "runs"
    / "run_001"
    / "generated_1000_wide"
    / "mf2_garch_regime_corr_1000x1260_wide.csv"
)

OUT_DIR = ROOT / "data" / "splits" / "mf2_garch_regime_corr"
WINDOW_LENGTH = 1260
STRIDE = 20
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20


def read_train_series(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    dates: list[str] = []
    values: list[tuple[str, str]] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                float(row["sp500"])
                float(row["DGS10"])
            except (KeyError, TypeError, ValueError):
                continue
            dates.append(row.get("date", ""))
            values.append((row["sp500"], row["DGS10"]))
    if len(values) < WINDOW_LENGTH:
        raise ValueError(f"too few train rows: {len(values)}")
    return dates, values


def real_windows(n_rows: int) -> list[tuple[int, int]]:
    return [
        (start, start + WINDOW_LENGTH - 1)
        for start in range(0, n_rows - WINDOW_LENGTH + 1, STRIDE)
    ]


def split_indices(n: int) -> dict[str, list[int]]:
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
    return {
        "train": list(range(0, train_end)),
        "val": list(range(train_end, val_end)),
        "test": list(range(val_end, n)),
    }


def real_windows_by_period(n_rows: int) -> tuple[list[tuple[int, int]], dict[str, list[int]], dict[str, tuple[int, int]]]:
    """Split raw time periods first, then create windows contained in each period."""
    train_end = int(n_rows * TRAIN_RATIO)
    val_end = int(n_rows * (TRAIN_RATIO + VAL_RATIO))
    periods = {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, n_rows),
    }
    windows: list[tuple[int, int]] = []
    split: dict[str, list[int]] = {}
    for name in ["train", "val", "test"]:
        period_start, period_end = periods[name]
        ids: list[int] = []
        for start in range(period_start, period_end - WINDOW_LENGTH + 1, STRIDE):
            ids.append(len(windows))
            windows.append((start, start + WINDOW_LENGTH - 1))
        split[name] = ids
    return windows, split, periods


def read_generated_wide(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [row for row in reader]
    if len(rows) != WINDOW_LENGTH:
        raise ValueError(f"generated rows must be {WINDOW_LENGTH}, got {len(rows)}")
    for row_no, row in enumerate(rows, start=2):
        if len(row) != len(header):
            raise ValueError(f"row {row_no} has {len(row)} columns, expected {len(header)}")

    column_index = {name: idx for idx, name in enumerate(header)}
    pairs: list[tuple[str, str]] = []
    for name in header:
        if name.endswith("_sp500"):
            counterpart = name[: -len("_sp500")] + "_DGS10"
            if counterpart in column_index:
                pairs.append((name, counterpart))
        elif name.startswith("sp500_path_"):
            counterpart = "dgs10_path_" + name[len("sp500_path_") :]
            if counterpart in column_index:
                pairs.append((name, counterpart))
    if not pairs:
        raise ValueError(f"no sp500/DGS10 generated column pairs found in {path}")

    normalized_header: list[str] = []
    normalized_rows = [[] for _ in rows]
    for sample_no, (sp_col, dg_col) in enumerate(pairs, start=1):
        normalized_header.extend(
            [f"sample{sample_no:04d}_sp500", f"sample{sample_no:04d}_DGS10"]
        )
        sp_idx = column_index[sp_col]
        dg_idx = column_index[dg_col]
        for row_no, row in enumerate(rows):
            normalized_rows[row_no].extend([row[sp_idx], row[dg_idx]])
    return normalized_header, normalized_rows


def write_labels(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "label",
        "source",
        "real_window_start",
        "real_window_end",
        "sp500_col",
        "DGS10_col",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_split_wide(
    split: str,
    real_ids: list[int],
    generated_ids: list[int],
    train_values: list[tuple[str, str]],
    windows: list[tuple[int, int]],
    generated_rows: list[list[str]],
    out_dir: Path,
) -> list[dict[str, object]]:
    labels: list[dict[str, object]] = []
    header: list[str] = []

    for idx in real_ids:
        sample_no = idx + 1
        sample_id = f"real{sample_no:04d}"
        sp_col = f"{sample_id}_sp500"
        dg_col = f"{sample_id}_DGS10"
        header.extend([sp_col, dg_col])
        start, end = windows[idx]
        labels.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": 1,
                "source": "real",
                "real_window_start": start,
                "real_window_end": end,
                "sp500_col": sp_col,
                "DGS10_col": dg_col,
            }
        )

    for idx in generated_ids:
        sample_no = idx + 1
        sample_id = f"gen{sample_no:04d}"
        sp_col = f"{sample_id}_sp500"
        dg_col = f"{sample_id}_DGS10"
        header.extend([sp_col, dg_col])
        labels.append(
            {
                "sample_id": sample_id,
                "split": split,
                "label": 0,
                "source": "generated",
                "real_window_start": "",
                "real_window_end": "",
                "sp500_col": sp_col,
                "DGS10_col": dg_col,
            }
        )

    out_path = out_dir / f"{split}_wide.csv"
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(WINDOW_LENGTH):
            row: list[str] = []
            for idx in real_ids:
                start, _ = windows[idx]
                sp, dg = train_values[start + t]
                row.extend([sp, dg])
            for idx in generated_ids:
                row.extend([generated_rows[t][2 * idx], generated_rows[t][2 * idx + 1]])
            writer.writerow(row)
    return labels


def main() -> None:
    global TRAIN_CSV, GENERATED_WIDE_CSV, OUT_DIR, WINDOW_LENGTH, STRIDE, TRAIN_RATIO, VAL_RATIO
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    TRAIN_CSV = (ROOT / config["train_csv"]).resolve()
    GENERATED_WIDE_CSV = (ROOT / config["generated_wide_csv"]).resolve()
    data_name = config.get("data_name", "mf2_garch_regime_corr")
    OUT_DIR = ROOT / config["experiment_dir"] / "data" / "splits" / data_name
    WINDOW_LENGTH = int(config["window_length"])
    STRIDE = int(config["stride"])
    TRAIN_RATIO = float(config["train_ratio"])
    VAL_RATIO = float(config["val_ratio"])
    dates, train_values = read_train_series(TRAIN_CSV)
    split_strategy = config.get("real_split_strategy", "window_then_split")
    if split_strategy == "period_then_window":
        windows, real_split, real_periods = real_windows_by_period(len(train_values))
    elif split_strategy == "window_then_split":
        windows = real_windows(len(train_values))
        real_split = split_indices(len(windows))
        real_periods = None
    else:
        raise ValueError(f"unknown real_split_strategy: {split_strategy}")
    gen_header, generated_rows = read_generated_wide(GENERATED_WIDE_CSV)
    n_generated = len(gen_header) // 2

    gen_split = split_indices(n_generated)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_labels: list[dict[str, object]] = []
    for split in ["train", "val", "test"]:
        split_labels = write_split_wide(
            split=split,
            real_ids=real_split[split],
            generated_ids=gen_split[split],
            train_values=train_values,
            windows=windows,
            generated_rows=generated_rows,
            out_dir=OUT_DIR,
        )
        write_labels(OUT_DIR / f"{split}_labels.csv", split_labels)
        all_labels.extend(split_labels)

    write_labels(OUT_DIR / "labels.csv", all_labels)

    summary = {
        "train_csv": str(TRAIN_CSV),
        "generated_wide_csv": str(GENERATED_WIDE_CSV),
        "output_dir": str(OUT_DIR),
        "window_length": WINDOW_LENGTH,
        "stride": STRIDE,
        "real_split_strategy": split_strategy,
        "real_windows": len(windows),
        "generated_samples": n_generated,
        "splits": {
            split: {
                "real": len(real_split[split]),
                "generated": len(gen_split[split]),
                "total": len(real_split[split]) + len(gen_split[split]),
                "wide_rows": WINDOW_LENGTH,
                "wide_columns": 2 * (len(real_split[split]) + len(gen_split[split])),
            }
            for split in ["train", "val", "test"]
        },
        "label_definition": {"real": 1, "generated": 0},
    }
    if real_periods is not None:
        summary["real_periods"] = {
            split: {
                "start_row": start,
                "end_row": end - 1,
                "start_date": dates[start],
                "end_date": dates[end - 1],
            }
            for split, (start, end) in real_periods.items()
        }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        "\n".join(
            [
                "# MF2 Regime-Correlation LightGBM Split Data",
                "",
                f"- real samples are {WINDOW_LENGTH}-day windows from train data with stride {STRIDE}.",
                f"- real split strategy: {split_strategy}.",
                "- period_then_window means raw periods are split first and no real window crosses a split boundary.",
                "- generated samples are split by sample order.",
                "- label: real=1, generated=0.",
                "",
                "## Split Counts",
                "",
                "| split | real | generated | total | wide columns |",
                "|---|---:|---:|---:|---:|",
                *[
                    f"| {split} | {summary['splits'][split]['real']} | {summary['splits'][split]['generated']} | {summary['splits'][split]['total']} | {summary['splits'][split]['wide_columns']} |"
                    for split in ["train", "val", "test"]
                ],
                "",
                "## Files",
                "",
                "- train_wide.csv, val_wide.csv, test_wide.csv",
                "- train_labels.csv, val_labels.csv, test_labels.csv",
                "- labels.csv",
                "- summary.json",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
