#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "mf2_garch_regime_corr_baseline"
FIG_DIR = RUN_DIR / "figures"
RESULT_DIR = RUN_DIR / "score_distribution"
SPLITS = ["train", "val", "test"]
BINS = 40


def read_predictions(split: str) -> list[dict[str, object]]:
    path = RUN_DIR / f"{split}_predictions.csv"
    rows: list[dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "source": row["source"],
                    "label": int(row["label"]),
                    "pred_label": int(row["pred_label"]),
                    "prob_real": float(row["prob_real"]),
                }
            )
    return rows


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = q * (len(ys) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ys) - 1)
    w = pos - lo
    return ys[lo] * (1 - w) + ys[hi] * w


def hist_counts(xs: list[float], bins: int = BINS) -> list[int]:
    counts = [0 for _ in range(bins)]
    for value in xs:
        idx = min(max(int(value * bins), 0), bins - 1)
        counts[idx] += 1
    return counts


def density(counts: list[int]) -> list[float]:
    total = sum(counts)
    return [c / total if total else 0.0 for c in counts]


def esc(text: object) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, width: int = 1200, height: int = 780):
        self.width = width
        self.height = height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
        ]

    def text(self, x: float, y: float, text: object, size: int = 18, anchor: str = "start", weight: str = "normal") -> None:
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-family="Helvetica,Arial,sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="#111827" text-anchor="{anchor}">{esc(text)}</text>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, color: str = "#111827", width: float = 1.5, dash: str = "") -> None:
        dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{width}"{dash_attr}/>')

    def rect(self, x: float, y: float, w: float, h: float, fill: str, opacity: float = 1.0, stroke: str = "none") -> None:
        self.parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="{fill}" '
            f'fill-opacity="{opacity:.3f}" stroke="{stroke}"/>'
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.parts.append("</svg>")
        path.write_text("\n".join(self.parts), encoding="utf-8")


def draw_hist_panel(svg: Svg, x0: float, y0: float, w: float, h: float, split: str, real_probs: list[float], gen_probs: list[float]) -> None:
    real_d = density(hist_counts(real_probs))
    gen_d = density(hist_counts(gen_probs))
    ymax = max(real_d + gen_d + [1e-9]) * 1.15
    svg.text(x0, y0 - 22, f"{split}: prob_real distribution", size=22, weight="bold")
    svg.line(x0, y0 + h, x0 + w, y0 + h, "#374151", 1.5)
    svg.line(x0, y0, x0, y0 + h, "#374151", 1.5)
    for i in range(6):
        x = x0 + w * i / 5
        svg.line(x, y0 + h, x, y0 + h + 6, "#374151", 1)
        svg.text(x, y0 + h + 25, f"{i / 5:.1f}", size=14, anchor="middle")
    for i in range(5):
        y = y0 + h - h * i / 4
        value = ymax * i / 4
        svg.line(x0 - 6, y, x0, y, "#374151", 1)
        svg.text(x0 - 10, y + 4, f"{value:.2f}", size=13, anchor="end")
    bar_w = w / BINS
    for idx, value in enumerate(gen_d):
        bh = h * value / ymax
        svg.rect(x0 + idx * bar_w, y0 + h - bh, bar_w * 0.92, bh, "#dc2626", opacity=0.42)
    for idx, value in enumerate(real_d):
        bh = h * value / ymax
        svg.rect(x0 + idx * bar_w + bar_w * 0.08, y0 + h - bh, bar_w * 0.72, bh, "#2563eb", opacity=0.42)
    threshold_x = x0 + w * 0.5
    svg.line(threshold_x, y0, threshold_x, y0 + h, "#111827", 1.5, dash="5,5")
    svg.text(threshold_x + 8, y0 + 18, "0.5", size=14)
    svg.text(x0 + w / 2, y0 + h + 52, "prob_real", size=15, anchor="middle")
    svg.text(x0 - 60, y0 + h / 2, "share per bin", size=15, anchor="middle")
    svg.rect(x0 + w - 180, y0 + 16, 18, 14, "#2563eb", opacity=0.42)
    svg.text(x0 + w - 154, y0 + 29, f"real n={len(real_probs)}", size=14)
    svg.rect(x0 + w - 180, y0 + 40, 18, 14, "#dc2626", opacity=0.42)
    svg.text(x0 + w - 154, y0 + 53, f"generated n={len(gen_probs)}", size=14)


def summarize_split(rows: list[dict[str, object]]) -> dict[str, object]:
    real = [float(row["prob_real"]) for row in rows if int(row["label"]) == 1]
    gen = [float(row["prob_real"]) for row in rows if int(row["label"]) == 0]
    return {
        "real_n": len(real),
        "generated_n": len(gen),
        "real_mean": sum(real) / len(real) if real else 0.0,
        "generated_mean": sum(gen) / len(gen) if gen else 0.0,
        "real_q05": quantile(real, 0.05),
        "real_q50": quantile(real, 0.50),
        "real_q95": quantile(real, 0.95),
        "generated_q05": quantile(gen, 0.05),
        "generated_q50": quantile(gen, 0.50),
        "generated_q95": quantile(gen, 0.95),
        "real_above_05": sum(1 for x in real if x >= 0.5) / len(real) if real else 0.0,
        "generated_below_05": sum(1 for x in gen if x < 0.5) / len(gen) if gen else 0.0,
    }


def write_summary(rows_by_split: dict[str, list[dict[str, object]]]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for split, rows_for_split in rows_by_split.items():
        row = {"split": split}
        row.update(summarize_split(rows_for_split))
        rows.append(row)
    with (RESULT_DIR / "score_distribution_summary.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (RESULT_DIR / "score_distribution_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def make_svg(rows_by_split: dict[str, list[dict[str, object]]]) -> Path:
    svg = Svg()
    svg.text(60, 40, "LightGBM score distribution", size=26, weight="bold")
    svg.text(60, 68, "Blue = real, red = generated, dashed line = default threshold 0.5", size=15)
    panel_w = 310
    panel_h = 420
    for idx, split in enumerate(SPLITS):
        rows = rows_by_split[split]
        real = [float(row["prob_real"]) for row in rows if int(row["label"]) == 1]
        gen = [float(row["prob_real"]) for row in rows if int(row["label"]) == 0]
        draw_hist_panel(svg, 90 + idx * 360, 145, panel_w, panel_h, split, real, gen)
        summary = summarize_split(rows)
        y = 640
        x = 90 + idx * 360
        svg.text(x, y, f"real median: {summary['real_q50']:.3f}", size=14)
        svg.text(x, y + 22, f"gen median: {summary['generated_q50']:.3f}", size=14)
        svg.text(x, y + 44, f"real >=0.5: {summary['real_above_05']:.1%}", size=14)
        svg.text(x, y + 66, f"gen <0.5: {summary['generated_below_05']:.1%}", size=14)
    path = FIG_DIR / "score_distribution.svg"
    svg.save(path)
    return path


def convert_to_pdf(svg_path: Path) -> Path | None:
    pdf_path = FIG_DIR / "score_distribution.pdf"
    env = os.environ.copy()
    env["MAGICK_TEMPORARY_PATH"] = "/private/tmp"
    env["MAGICK_OCL_DEVICE"] = "off"
    try:
        subprocess.run(["convert", str(svg_path), str(pdf_path)], check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        (RESULT_DIR / "pdf_error.txt").write_text(str(exc), encoding="utf-8")
        return None
    return pdf_path


def main() -> None:
    global RUN_DIR, FIG_DIR, RESULT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = ROOT / config["experiment_dir"]
    RUN_DIR = experiment_dir / "results"
    FIG_DIR = experiment_dir / "figures"
    RESULT_DIR = RUN_DIR / "score_distribution"
    rows_by_split = {split: read_predictions(split) for split in SPLITS}
    write_summary(rows_by_split)
    svg_path = make_svg(rows_by_split)
    pdf_path = convert_to_pdf(svg_path)
    print(f"summary: {RESULT_DIR / 'score_distribution_summary.csv'}")
    print(f"svg: {svg_path}")
    if pdf_path:
        print(f"pdf: {pdf_path}")
    for split in SPLITS:
        s = summarize_split(rows_by_split[split])
        print(
            f"{split}: real_median={s['real_q50']:.4f}, generated_median={s['generated_q50']:.4f}, "
            f"real>=0.5={s['real_above_05']:.2%}, generated<0.5={s['generated_below_05']:.2%}"
        )


if __name__ == "__main__":
    main()
