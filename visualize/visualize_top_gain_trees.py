#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import pickle
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "common"))

from build_features import FEATURE_COLUMNS  # noqa: E402


def tree_gain(node: dict) -> float:
    if "leaf_index" in node:
        return 0.0
    return (
        float(node.get("split_gain", 0.0))
        + tree_gain(node["left_child"])
        + tree_gain(node["right_child"])
    )


def leaf_count(node: dict) -> int:
    if "leaf_index" in node:
        return 1
    return leaf_count(node["left_child"]) + leaf_count(node["right_child"])


def max_depth(node: dict, depth: int = 0) -> int:
    if "leaf_index" in node:
        return depth
    return max(
        max_depth(node["left_child"], depth + 1),
        max_depth(node["right_child"], depth + 1),
    )


def assign_positions(node: dict, depth: int, next_leaf: list[int], positions: dict[int, tuple[float, int]]) -> float:
    node_id = id(node)
    if "leaf_index" in node:
        x = float(next_leaf[0])
        next_leaf[0] += 1
        positions[node_id] = (x, depth)
        return x
    left_x = assign_positions(node["left_child"], depth + 1, next_leaf, positions)
    right_x = assign_positions(node["right_child"], depth + 1, next_leaf, positions)
    x = (left_x + right_x) / 2.0
    positions[node_id] = (x, depth)
    return x


def original_threshold(threshold: float, feature_index: int, scaler) -> float:
    return threshold * float(scaler.scale_[feature_index]) + float(scaler.mean_[feature_index])


def render_tree_svg(
    tree: dict,
    tree_index: int,
    gain_total: float,
    feature_names: list[str],
    scaler,
    output_path: Path,
) -> dict:
    structure = tree["tree_structure"]
    leaves = leaf_count(structure)
    depth = max_depth(structure)
    x_step = 205
    y_step = 170
    margin_x = 125
    margin_top = 105
    width = max(1250, margin_x * 2 + max(leaves - 1, 1) * x_step)
    height = margin_top + (depth + 1) * y_step + 80
    positions: dict[int, tuple[float, int]] = {}
    assign_positions(structure, 0, [0], positions)

    def canvas_position(node: dict) -> tuple[float, float]:
        leaf_x, node_depth = positions[id(node)]
        if leaves == 1:
            x = width / 2
        else:
            x = margin_x + leaf_x * (width - 2 * margin_x) / (leaves - 1)
        return x, margin_top + node_depth * y_step

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.title{font-size:25px;font-weight:700;fill:#0f172a}.sub{font-size:15px;fill:#475569}.node-title{font-size:14px;font-weight:700;fill:#0f172a}.node-text{font-size:12px;fill:#334155}.edge{stroke:#94a3b8;stroke-width:2}.edge-label{font-size:12px;font-weight:700;fill:#475569}</style>',
        f'<text x="30" y="38" class="title">Tree {tree_index} — gain合計 {gain_total:.6g}</text>',
        f'<text x="30" y="66" class="sub">葉数 {leaves} / 最大深さ {depth} / 左枝: 条件成立 / 右枝: 条件不成立</text>',
    ]

    def edges(node: dict) -> None:
        if "leaf_index" in node:
            return
        x1, y1 = canvas_position(node)
        for child, label in [(node["left_child"], "Yes"), (node["right_child"], "No")]:
            x2, y2 = canvas_position(child)
            parts.append(f'<line x1="{x1:.1f}" y1="{y1 + 44:.1f}" x2="{x2:.1f}" y2="{y2 - 44:.1f}" class="edge"/>')
            parts.append(f'<text x="{(x1 + x2) / 2:.1f}" y="{(y1 + y2) / 2 - 5:.1f}" text-anchor="middle" class="edge-label">{label}</text>')
            edges(child)

    edges(structure)

    used_features: dict[str, dict[str, float | int]] = {}

    def nodes(node: dict) -> None:
        x, y = canvas_position(node)
        box_w, box_h = 190, 88
        if "leaf_index" in node:
            fill = "#dcfce7" if float(node["leaf_value"]) >= 0 else "#fee2e2"
            parts.append(f'<rect x="{x - box_w/2:.1f}" y="{y - box_h/2:.1f}" width="{box_w}" height="{box_h}" rx="10" fill="{fill}" stroke="#64748b"/>')
            lines = [
                (f'Leaf {node["leaf_index"]}', "node-title"),
                (f'値 {float(node["leaf_value"]):.5g}', "node-text"),
                (f'件数 {int(node.get("leaf_count", 0))}', "node-text"),
            ]
        else:
            feature_index = int(node["split_feature"])
            feature = feature_names[feature_index]
            threshold = float(node["threshold"])
            raw_threshold = original_threshold(threshold, feature_index, scaler)
            gain = float(node.get("split_gain", 0.0))
            stats = used_features.setdefault(feature, {"split_count": 0, "gain": 0.0})
            stats["split_count"] = int(stats["split_count"]) + 1
            stats["gain"] = float(stats["gain"]) + gain
            parts.append(f'<rect x="{x - box_w/2:.1f}" y="{y - box_h/2:.1f}" width="{box_w}" height="{box_h}" rx="10" fill="#dbeafe" stroke="#2563eb"/>')
            lines = [
                (html.escape(feature), "node-title"),
                (f'標準化後 ≤ {threshold:.5g}', "node-text"),
                (f'元尺度 ≤ {raw_threshold:.5g}', "node-text"),
                (f'gain {gain:.5g} / 件数 {int(node.get("internal_count", 0))}', "node-text"),
            ]
        start_y = y - 24
        for line_no, (text_value, css_class) in enumerate(lines):
            parts.append(f'<text x="{x:.1f}" y="{start_y + 19*line_no:.1f}" text-anchor="middle" class="{css_class}">{text_value}</text>')
        if "leaf_index" not in node:
            nodes(node["left_child"])
            nodes(node["right_child"])

    nodes(structure)
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
    return {
        "tree_index": tree_index,
        "gain_total": gain_total,
        "num_leaves": leaves,
        "max_depth": depth,
        "used_features": used_features,
        "svg": output_path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBMのgain上位の木をSVG化する")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=ROOT / "train/stride20_blocked_cv_za_mf2/model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "visualize/stride20_blocked_cv_za_mf2/top_gain_trees",
    )
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with (args.model_dir / "lgb_model.pkl").open("rb") as f:
        model = pickle.load(f)
    with (args.model_dir / "scaler.pkl").open("rb") as f:
        scaler = pickle.load(f)

    dump = model.booster_.dump_model()
    dumped_feature_names = list(dump["feature_names"])
    feature_names = (
        list(FEATURE_COLUMNS)
        if len(dumped_feature_names) == len(FEATURE_COLUMNS)
        else dumped_feature_names
    )
    ranked = sorted(
        [
            (index, tree_gain(tree["tree_structure"]), tree)
            for index, tree in enumerate(dump["tree_info"])
        ],
        key=lambda item: item[1],
        reverse=True,
    )
    selected = ranked[: args.top_k]
    results = []
    for rank, (tree_index, gain_total, tree) in enumerate(selected, start=1):
        output_path = args.output_dir / f"rank_{rank:02d}_tree_{tree_index:02d}.svg"
        result = render_tree_svg(
            tree, tree_index, gain_total, feature_names, scaler, output_path
        )
        result["rank"] = rank
        results.append(result)

    summary = {
        "model_dir": str(args.model_dir),
        "tree_count": len(dump["tree_info"]),
        "ranking_metric": "sum of split_gain in each tree",
        "top_k": args.top_k,
        "trees": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
