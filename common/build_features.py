#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "splits" / "mf2_garch_regime_corr"
OUT_DIR = ROOT / "data" / "features" / "mf2_garch_regime_corr_baseline"


FEATURE_COLUMNS = [
    "sp_mean",
    "sp_std",
    "sp_skew",
    "sp_kurt",
    "sp_q90_abs",
    "sp_q95_abs",
    "sp_q99_abs",
    "rate_mean",
    "rate_std",
    "rate_skew",
    "rate_kurt",
    "rate_q90_abs",
    "rate_q95_abs",
    "rate_q99_abs",
    "sp_acf1",
    "sp_acf5",
    "sp_abs_acf1",
    "sp_abs_acf5",
    "rate_acf1",
    "rate_acf5",
    "rate_abs_acf1",
    "rate_abs_acf5",
    "corr_sp_rate",
    "joint_large_move_95",
    "sp_tail_ratio",
    "rate_tail_ratio",
]


def read_wide(path: Path) -> tuple[list[str], list[list[float]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [[float(x) for x in row] for row in reader]
    return header, rows


def read_labels(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = mean(xs)
    return math.sqrt(max(sum((x - mu) ** 2 for x in xs) / (len(xs) - 1), 0.0))


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    w = pos - lo
    return ys[lo] * (1.0 - w) + ys[hi] * w


def skew(xs: list[float]) -> float:
    sigma = std(xs)
    if len(xs) < 3 or sigma <= 0.0:
        return 0.0
    mu = mean(xs)
    return sum(((x - mu) / sigma) ** 3 for x in xs) / len(xs)


def kurtosis_excess(xs: list[float]) -> float:
    sigma = std(xs)
    if len(xs) < 4 or sigma <= 0.0:
        return 0.0
    mu = mean(xs)
    return sum(((x - mu) / sigma) ** 4 for x in xs) / len(xs) - 3.0


def corr(xs: list[float], ys: list[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 3:
        return 0.0
    x = xs[:n]
    y = ys[:n]
    mx = mean(x)
    my = mean(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den_x = sum((a - mx) ** 2 for a in x)
    den_y = sum((b - my) ** 2 for b in y)
    den = math.sqrt(den_x * den_y)
    return num / den if den > 0.0 else 0.0


def autocorr(xs: list[float], lag: int) -> float:
    if len(xs) <= lag:
        return 0.0
    return corr(xs[:-lag], xs[lag:])


def features_for_pair(sp: list[float], rate: list[float]) -> dict[str, float]:
    sp_abs = [abs(x) for x in sp]
    rate_abs = [abs(x) for x in rate]
    sp_std = std(sp)
    rate_std = std(rate)
    sp_q90 = quantile(sp_abs, 0.90)
    sp_q95 = quantile(sp_abs, 0.95)
    sp_q99 = quantile(sp_abs, 0.99)
    rate_q90 = quantile(rate_abs, 0.90)
    rate_q95 = quantile(rate_abs, 0.95)
    rate_q99 = quantile(rate_abs, 0.99)
    return {
        "sp_mean": mean(sp),
        "sp_std": sp_std,
        "sp_skew": skew(sp),
        "sp_kurt": kurtosis_excess(sp),
        "sp_q90_abs": sp_q90,
        "sp_q95_abs": sp_q95,
        "sp_q99_abs": sp_q99,
        "rate_mean": mean(rate),
        "rate_std": rate_std,
        "rate_skew": skew(rate),
        "rate_kurt": kurtosis_excess(rate),
        "rate_q90_abs": rate_q90,
        "rate_q95_abs": rate_q95,
        "rate_q99_abs": rate_q99,
        "sp_acf1": autocorr(sp, 1),
        "sp_acf5": autocorr(sp, 5),
        "sp_abs_acf1": autocorr(sp_abs, 1),
        "sp_abs_acf5": autocorr(sp_abs, 5),
        "rate_acf1": autocorr(rate, 1),
        "rate_acf5": autocorr(rate, 5),
        "rate_abs_acf1": autocorr(rate_abs, 1),
        "rate_abs_acf5": autocorr(rate_abs, 5),
        "corr_sp_rate": corr(sp, rate),
        "joint_large_move_95": sum(
            1 for a, b in zip(sp_abs, rate_abs) if a > sp_q95 and b > rate_q95
        ) / len(sp_abs),
        "sp_tail_ratio": sp_q99 / sp_std if sp_std != 0.0 else 0.0,
        "rate_tail_ratio": rate_q99 / rate_std if rate_std != 0.0 else 0.0,
    }


def build_split(split: str) -> list[dict[str, object]]:
    header, rows = read_wide(SPLIT_DIR / f"{split}_wide.csv")
    labels = read_labels(SPLIT_DIR / f"{split}_labels.csv")
    col_index = {name: idx for idx, name in enumerate(header)}
    out: list[dict[str, object]] = []
    for label_row in labels:
        sp_col = label_row["sp500_col"]
        rate_col = label_row["DGS10_col"]
        sp_idx = col_index[sp_col]
        rate_idx = col_index[rate_col]
        sp = [row[sp_idx] for row in rows]
        rate = [row[rate_idx] for row in rows]
        feature_row: dict[str, object] = {
            "sample_id": label_row["sample_id"],
            "split": split,
            "label": int(label_row["label"]),
            "source": label_row["source"],
        }
        feature_row.update(features_for_pair(sp, rate))
        out.append(feature_row)
    return out


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "split", "label", "source", *FEATURE_COLUMNS]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    global SPLIT_DIR, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = ROOT / config["experiment_dir"]
    SPLIT_DIR = experiment_dir / "data" / "splits" / "mf2_garch_regime_corr"
    OUT_DIR = experiment_dir / "data" / "features"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    summary: dict[str, object] = {"feature_count": len(FEATURE_COLUMNS), "features": FEATURE_COLUMNS, "splits": {}}
    for split in ["train", "val", "test"]:
        rows = build_split(split)
        write_rows(OUT_DIR / f"{split}_features.csv", rows)
        all_rows.extend(rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "real": sum(1 for row in rows if row["label"] == 1),
            "generated": sum(1 for row in rows if row["label"] == 0),
        }
    write_rows(OUT_DIR / "all_features.csv", all_rows)
    (OUT_DIR / "feature_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
