#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]


def load_pairs(path: Path, metadata_columns: int) -> tuple[list[str], list[list[str]], np.ndarray]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    numeric = np.asarray(
        [[float(value) for value in row[metadata_columns:]] for row in rows], dtype=float
    )
    pair_count = numeric.shape[1] // 2
    pairs = numeric.reshape(len(rows), pair_count, 2).transpose(1, 0, 2)
    return header, rows, pairs


def digest(pair: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pair).tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="combined内でZA・MF2と一致しない系列を抽出する")
    parser.add_argument("--combined-part1", type=Path, default=ROOT / "combined_1000paths_wide_part1.csv")
    parser.add_argument("--combined-part2", type=Path, default=ROOT / "combined_1000paths_wide_part2.csv")
    parser.add_argument("--za", type=Path, default=ROOT / "za_final7_1000paths_wide.csv")
    parser.add_argument(
        "--mf2",
        type=Path,
        default=ROOT / "garch_origin/models/mf2_garch_regime_corr/runs/run_001/generated_1000_wide/mf2_garch_regime_corr_1000x1260_wide.csv",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "sigwgan_120paths_wide.csv")
    args = parser.parse_args()

    combined_parts = []
    combined_raw = []
    combined_dates: list[str] | None = None
    for path in [args.combined_part1, args.combined_part2]:
        _, rows, pairs = load_pairs(path, metadata_columns=1)
        dates = [row[0] for row in rows]
        if combined_dates is None:
            combined_dates = dates
        elif dates != combined_dates:
            raise ValueError("combined part1とpart2の日付列が一致しない")
        combined_parts.append(pairs)
        combined_raw.append(rows)

    combined_pairs = np.concatenate(combined_parts, axis=0)
    _, _, za_pairs = load_pairs(args.za, metadata_columns=2)
    _, _, mf2_pairs = load_pairs(args.mf2, metadata_columns=0)
    known = {digest(pair) for pair in za_pairs} | {digest(pair) for pair in mf2_pairs}
    unmatched = [index for index, pair in enumerate(combined_pairs) if digest(pair) not in known]
    if len(unmatched) != 120:
        raise ValueError(f"ZA・MF2と不一致の系列は120件の想定だが、{len(unmatched)}件だった")

    output_header = ["date"]
    for new_index in range(len(unmatched)):
        output_header.extend([f"sp500_path_{new_index}", f"dgs10_path_{new_index}"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(output_header)
        for row_index, date in enumerate(combined_dates or []):
            output_row = [date]
            for original_index in unmatched:
                part_no = 0 if original_index < 500 else 1
                local_index = original_index if part_no == 0 else original_index - 500
                source_row = combined_raw[part_no][row_index]
                output_row.extend(
                    [source_row[1 + 2 * local_index], source_row[1 + 2 * local_index + 1]]
                )
            writer.writerow(output_row)

    manifest = {
        "output_csv": str(args.output),
        "source_files": [str(args.combined_part1), str(args.combined_part2)],
        "excluded_reference_files": {"za": str(args.za), "mf2": str(args.mf2)},
        "selection_rule": "1260x2 float values do not exactly match any ZA or MF2 pair",
        "assigned_source": "sigwgan",
        "n_rows": len(combined_dates or []),
        "n_pairs": len(unmatched),
        "original_combined_indices_zero_based": unmatched,
        "output_pair_mapping": [
            {"output_path_index": new_index, "combined_path_index": original_index}
            for new_index, original_index in enumerate(unmatched)
        ],
    }
    manifest_path = args.output.with_name(args.output.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
