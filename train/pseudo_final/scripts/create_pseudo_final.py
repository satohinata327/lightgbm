#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


WINDOW_LENGTH = 1260


def read_generated(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    if len(rows) != WINDOW_LENGTH or len(header) % 2:
        raise ValueError(f"expected 1260 rows and paired columns: {path}")
    return header, rows


def read_real(path: Path) -> list[dict[str, str]]:
    usable = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                float(row["sp500"])
                float(row["DGS10"])
            except (KeyError, TypeError, ValueError):
                continue
            usable.append(row)
    return usable


def choose_nonoverlapping_starts(rng: random.Random, n_rows: int) -> list[int]:
    starts = list(range(n_rows - WINDOW_LENGTH + 1))
    first = rng.choice(starts)
    candidates = [
        start
        for start in starts
        if start + WINDOW_LENGTH <= first or first + WINDOW_LENGTH <= start
    ]
    if not candidates:
        raise ValueError("train data is too short for two non-overlapping windows")
    return [first, rng.choice(candidates)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    generated_header, generated_rows = read_generated(args.generated)
    real_rows = read_real(args.train)
    n_generated = len(generated_header) // 2
    generated_ids = rng.sample(range(n_generated), 12)
    real_starts = choose_nonoverlapping_starts(rng, len(real_rows))

    header = []
    for mask_no in range(1, 15):
        header.extend([f"mask{mask_no}_sp500", f"mask{mask_no}_DGS10"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for t in range(WINDOW_LENGTH):
            out = []
            for generated_idx in generated_ids:
                out.extend(generated_rows[t][2 * generated_idx : 2 * generated_idx + 2])
            for start in real_starts:
                row = real_rows[start + t]
                out.extend([row["sp500"], row["DGS10"]])
            writer.writerow(out)

    manifest = {
        "seed": args.seed,
        "window_length": WINDOW_LENGTH,
        "generated_source": str(args.generated),
        "real_source": str(args.train),
        "masks": {},
    }
    for mask_no, generated_idx in enumerate(generated_ids, start=1):
        manifest["masks"][f"mask{mask_no}"] = {
            "label": 0,
            "source": "generated",
            "source_sample": f"sample{generated_idx + 1:04d}",
        }
    for offset, start in enumerate(real_starts, start=13):
        manifest["masks"][f"mask{offset}"] = {
            "label": 1,
            "source": "real",
            "start_row": start,
            "end_row": start + WINDOW_LENGTH - 1,
            "start_date": real_rows[start].get("date", ""),
            "end_date": real_rows[start + WINDOW_LENGTH - 1].get("date", ""),
        }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
