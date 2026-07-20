#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "final_model/ensemble_baseline_no_joint_v1/ensemble_config.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="最終アンサンブルモデルでmixed形式CSVを判定する"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "common/score_ensemble_mixed_csv.py"),
            "--config",
            str(DEFAULT_CONFIG),
            "--input",
            str(args.input.resolve()),
            "--output-dir",
            str(args.output_dir.resolve()),
        ],
        cwd=ROOT,
        check=True,
    )


if __name__ == "__main__":
    main()
