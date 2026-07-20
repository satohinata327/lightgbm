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

JOINT_EQUAL_MEAN_FEATURE = "joint_large_move_90_99_mean"
JOINT_SMOOTH_RANK_FEATURE = "joint_large_move_rank95_smooth_tau02"
JOINT_DIAGNOSTIC_COLUMNS = [f"joint_large_move_{q}" for q in range(90, 100)]
JOINT_DIRECTIONAL_FEATURES = [
    "joint_large_move_95_sp_pos_rate_pos",
    "joint_large_move_95_sp_pos_rate_neg",
    "joint_large_move_95_sp_neg_rate_pos",
    "joint_large_move_95_sp_neg_rate_neg",
]
VOL_DOWNSIDE_FEATURES = [
    "vol_abs_corr_coupling_60",
    "high_vol_abs_corr_gap_60",
    "sp_downside_corr_gap_10",
    "sp_downside_abs_corr_gap_10",
]

UNIVARIATE_STYLIZED_14_FEATURES = [
    "sp_mean",
    "sp_std",
    "sp_skew",
    "sp_kurt",
    "sp_abs_acf1",
    "sp_abs_future_vol_corr_5",
    "sp_leverage_corr_1",
    "sp_leverage_corr_5",
    "rate_mean",
    "rate_std",
    "rate_skew",
    "rate_kurt",
    "rate_abs_acf1",
    "rate_abs_future_vol_corr_5",
]

UNIVARIATE_STYLIZED_12_NO_STD_FEATURES = [
    feature
    for feature in UNIVARIATE_STYLIZED_14_FEATURES
    if feature not in {"sp_std", "rate_std"}
]

CROSS_ASSET_16_FEATURES = [
    "rolling_corr60_std", "rolling_corr60_q10", "rolling_corr60_q50",
    "rolling_corr60_q90", "corr_sp_rate_given_sp_down_q10",
    "corr_sp_rate_given_sp_up_q90", "corr_asymmetry_sp_down_vs_up",
    "corr_abs_sp_rate", "corr_rolling_vol20_sp_rate", "joint_high_vol20_q90",
    "sp_lead_rate_corr_1", "rate_lead_sp_corr_1",
    "sp_lead_rate_next5_sum_corr", "rate_lead_sp_next5_sum_corr",
    "sp_vol_lead_rate_vol_corr_1", "rate_vol_lead_sp_vol_corr_1",
]

CROSS_ASSET_17_WITH_JOINT95_FEATURES = [
    *CROSS_ASSET_16_FEATURES,
    "joint_large_move_95",
]

CROSS_ASSET_15_JOINT95_NO_ABS_CORR_NO_RATE_LEAD1_FEATURES = [
    feature
    for feature in CROSS_ASSET_17_WITH_JOINT95_FEATURES
    if feature not in {"corr_abs_sp_rate", "rate_lead_sp_corr_1"}
]

CROSS_ASSET_11_JOINT95_NO_ROLLING_CORR60_FEATURES = [
    feature
    for feature in CROSS_ASSET_15_JOINT95_NO_ABS_CORR_NO_RATE_LEAD1_FEATURES
    if feature
    not in {
        "rolling_corr60_std",
        "rolling_corr60_q10",
        "rolling_corr60_q50",
        "rolling_corr60_q90",
    }
]

CROSS_ASSET_10_NO_JOINT_NO_ROLLING_CORR60_FEATURES = [
    feature
    for feature in CROSS_ASSET_11_JOINT95_NO_ROLLING_CORR60_FEATURES
    if feature != "joint_large_move_95"
]

CROSS_ASSET_11_SMOOTH_JOINT_NO_ROLLING_CORR60_FEATURES = [
    *CROSS_ASSET_10_NO_JOINT_NO_ROLLING_CORR60_FEATURES,
    JOINT_SMOOTH_RANK_FEATURE,
]

SMOOTH_JOINT_CROSS_ASSET_11_PLUS_UNIVARIATE_14_FEATURES = [
    *CROSS_ASSET_11_SMOOTH_JOINT_NO_ROLLING_CORR60_FEATURES,
    *UNIVARIATE_STYLIZED_14_FEATURES,
]

NO_JOINT_CROSS_ASSET_10_PLUS_UNIVARIATE_14_FEATURES = [
    *CROSS_ASSET_10_NO_JOINT_NO_ROLLING_CORR60_FEATURES,
    *UNIVARIATE_STYLIZED_14_FEATURES,
]

BASELINE_SMOOTH_JOINT_PLUS_SP_LEAD_RATE_CORR1_FEATURES = [
    *[
        JOINT_SMOOTH_RANK_FEATURE if feature == "joint_large_move_95" else feature
        for feature in FEATURE_COLUMNS
    ],
    "sp_lead_rate_corr_1",
]

HYBRID_18_FEATURES = [
    "sp_std", "sp_skew", "sp_kurt", "sp_abs_acf1",
    "sp_leverage_corr_1", "sp_leverage_corr_5",
    "rate_std", "rate_abs_acf1", "rate_abs_future_vol_corr_5",
    "sp_lead_rate_corr_1", "rate_lead_sp_corr_1",
    "sp_lead_rate_next5_sum_corr", "rate_lead_sp_next5_sum_corr",
    "rolling_corr60_std", "rolling_corr60_q50", "rolling_corr60_q90",
    "corr_abs_sp_rate", "corr_rolling_vol20_sp_rate",
]


def rolling_corr_feature_columns(
    window: int, statistics: list[str] | None = None
) -> list[str]:
    prefix = f"rolling_corr_{window}"
    if statistics is None:
        statistics = [
            "mean", "std", "q10", "q50", "q90", "min", "max", "negative_fraction"
        ]
    return [f"{prefix}_{statistic}" for statistic in statistics]


def feature_columns(
    rolling_corr_window: int | None = None,
    rolling_corr_statistics: list[str] | None = None,
    joint_feature_mode: str = "single_95",
    dependence_feature_mode: str = "none",
    feature_set_mode: str = "baseline",
) -> list[str]:
    if feature_set_mode == "univariate_stylized_14":
        return list(UNIVARIATE_STYLIZED_14_FEATURES)
    if feature_set_mode == "univariate_stylized_12_no_std":
        return list(UNIVARIATE_STYLIZED_12_NO_STD_FEATURES)
    if feature_set_mode == "cross_asset_16":
        return list(CROSS_ASSET_16_FEATURES)
    if feature_set_mode == "cross_asset_17_with_joint95":
        return list(CROSS_ASSET_17_WITH_JOINT95_FEATURES)
    if feature_set_mode == "cross_asset_15_joint95_no_abs_corr_no_rate_lead1":
        return list(CROSS_ASSET_15_JOINT95_NO_ABS_CORR_NO_RATE_LEAD1_FEATURES)
    if feature_set_mode == "cross_asset_11_joint95_no_rolling_corr60":
        return list(CROSS_ASSET_11_JOINT95_NO_ROLLING_CORR60_FEATURES)
    if feature_set_mode == "cross_asset_10_no_joint_no_rolling_corr60":
        return list(CROSS_ASSET_10_NO_JOINT_NO_ROLLING_CORR60_FEATURES)
    if feature_set_mode == "cross_asset_11_smooth_joint_no_rolling_corr60":
        return list(CROSS_ASSET_11_SMOOTH_JOINT_NO_ROLLING_CORR60_FEATURES)
    if feature_set_mode == "smooth_joint_cross_asset_11_plus_univariate_14":
        return list(SMOOTH_JOINT_CROSS_ASSET_11_PLUS_UNIVARIATE_14_FEATURES)
    if feature_set_mode == "no_joint_cross_asset_10_plus_univariate_14":
        return list(NO_JOINT_CROSS_ASSET_10_PLUS_UNIVARIATE_14_FEATURES)
    if feature_set_mode == "baseline_smooth_joint_plus_sp_lead_rate_corr1":
        return list(BASELINE_SMOOTH_JOINT_PLUS_SP_LEAD_RATE_CORR1_FEATURES)
    if feature_set_mode == "hybrid_18":
        return list(HYBRID_18_FEATURES)
    if feature_set_mode != "baseline":
        raise ValueError(f"Unknown feature_set_mode: {feature_set_mode}")
    columns = list(FEATURE_COLUMNS)
    if joint_feature_mode == "equal_mean_90_99":
        joint_index = columns.index("joint_large_move_95")
        columns[joint_index] = JOINT_EQUAL_MEAN_FEATURE
    elif joint_feature_mode == "directional_95":
        joint_index = columns.index("joint_large_move_95")
        columns[joint_index : joint_index + 1] = JOINT_DIRECTIONAL_FEATURES
    elif joint_feature_mode == "none":
        columns.remove("joint_large_move_95")
    elif joint_feature_mode == "smooth_rank_95_tau02":
        joint_index = columns.index("joint_large_move_95")
        columns[joint_index] = JOINT_SMOOTH_RANK_FEATURE
    elif joint_feature_mode != "single_95":
        raise ValueError(f"Unknown joint_feature_mode: {joint_feature_mode}")
    if rolling_corr_window is not None:
        columns.extend(
            rolling_corr_feature_columns(rolling_corr_window, rolling_corr_statistics)
        )
    if dependence_feature_mode == "vol_downside_60_q10":
        columns.extend(VOL_DOWNSIDE_FEATURES)
    elif dependence_feature_mode != "none":
        raise ValueError(
            f"Unknown dependence_feature_mode: {dependence_feature_mode}"
        )
    return columns


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


def leverage_corr(xs: list[float], horizon: int) -> float:
    """Correlation between today's return and subsequent realized volatility."""
    if horizon < 1 or len(xs) <= horizon:
        return 0.0
    future_vol = [
        math.sqrt(mean([value * value for value in xs[t + 1 : t + horizon + 1]]))
        for t in range(len(xs) - horizon)
    ]
    return corr(xs[: len(future_vol)], future_vol)


def abs_future_vol_corr(xs: list[float], horizon: int) -> float:
    """Correlation between today's absolute move and subsequent realized volatility."""
    if horizon < 1 or len(xs) <= horizon:
        return 0.0
    future_vol = [
        math.sqrt(mean([value * value for value in xs[t + 1 : t + horizon + 1]]))
        for t in range(len(xs) - horizon)
    ]
    return corr([abs(value) for value in xs[: len(future_vol)]], future_vol)


def rolling_corr(xs: list[float], ys: list[float], window: int) -> list[float]:
    n = min(len(xs), len(ys))
    if window < 3 or n < window:
        return []
    x = xs[:n]
    y = ys[:n]
    sx = sum(x[:window])
    sy = sum(y[:window])
    sxx = sum(value * value for value in x[:window])
    syy = sum(value * value for value in y[:window])
    sxy = sum(a * b for a, b in zip(x[:window], y[:window]))
    values: list[float] = []
    for start in range(n - window + 1):
        if start > 0:
            old_x, old_y = x[start - 1], y[start - 1]
            new_x, new_y = x[start + window - 1], y[start + window - 1]
            sx += new_x - old_x
            sy += new_y - old_y
            sxx += new_x * new_x - old_x * old_x
            syy += new_y * new_y - old_y * old_y
            sxy += new_x * new_y - old_x * old_y
        numerator = sxy - sx * sy / window
        denominator = math.sqrt(
            max(sxx - sx * sx / window, 0.0) * max(syy - sy * sy / window, 0.0)
        )
        values.append(numerator / denominator if denominator > 0.0 else 0.0)
    return values


def rolling_std(xs: list[float], window: int) -> list[float]:
    if window < 2 or len(xs) < window:
        return []
    sx = sum(xs[:window])
    sxx = sum(value * value for value in xs[:window])
    values: list[float] = []
    for start in range(len(xs) - window + 1):
        if start > 0:
            old_value = xs[start - 1]
            new_value = xs[start + window - 1]
            sx += new_value - old_value
            sxx += new_value * new_value - old_value * old_value
        variance = max((sxx - sx * sx / window) / (window - 1), 0.0)
        values.append(math.sqrt(variance))
    return values


def rolling_realized_vol(xs: list[float], window: int) -> list[float]:
    if window < 1 or len(xs) < window:
        return []
    squared = [value * value for value in xs]
    total = sum(squared[:window])
    values = [math.sqrt(total / window)]
    for end in range(window, len(xs)):
        total += squared[end] - squared[end - window]
        values.append(math.sqrt(max(total / window, 0.0)))
    return values


def cross_asset_features(sp: list[float], rate: list[float]) -> dict[str, float]:
    n = min(len(sp), len(rate))
    x, y = sp[:n], rate[:n]
    rolling_corr60 = rolling_corr(x, y, 60)
    sp_q10, sp_q90 = quantile(x, 0.10), quantile(x, 0.90)
    down = [(a, b) for a, b in zip(x, y) if a < sp_q10]
    up = [(a, b) for a, b in zip(x, y) if a > sp_q90]
    corr_down = corr([a for a, _ in down], [b for _, b in down])
    corr_up = corr([a for a, _ in up], [b for _, b in up])

    sp_vol20 = rolling_realized_vol(x, 20)
    rate_vol20 = rolling_realized_vol(y, 20)
    vol_n = min(len(sp_vol20), len(rate_vol20))
    sp_vol20, rate_vol20 = sp_vol20[:vol_n], rate_vol20[:vol_n]
    sp_vol_q90 = quantile(sp_vol20, 0.90)
    rate_vol_q90 = quantile(rate_vol20, 0.90)
    joint_high_vol = (
        sum(a > sp_vol_q90 and b > rate_vol_q90 for a, b in zip(sp_vol20, rate_vol20)) / vol_n
        if vol_n else 0.0
    )

    lead_n = max(n - 5, 0)
    rate_next5 = [sum(y[t + 1:t + 6]) for t in range(lead_n)]
    sp_next5 = [sum(x[t + 1:t + 6]) for t in range(lead_n)]
    return {
        "rolling_corr60_std": std(rolling_corr60),
        "rolling_corr60_q10": quantile(rolling_corr60, 0.10),
        "rolling_corr60_q50": quantile(rolling_corr60, 0.50),
        "rolling_corr60_q90": quantile(rolling_corr60, 0.90),
        "corr_sp_rate_given_sp_down_q10": corr_down,
        "corr_sp_rate_given_sp_up_q90": corr_up,
        "corr_asymmetry_sp_down_vs_up": corr_down - corr_up,
        "corr_abs_sp_rate": corr([abs(value) for value in x], [abs(value) for value in y]),
        "corr_rolling_vol20_sp_rate": corr(sp_vol20, rate_vol20),
        "joint_high_vol20_q90": joint_high_vol,
        "sp_lead_rate_corr_1": corr(x[:-1], y[1:]) if n > 1 else 0.0,
        "rate_lead_sp_corr_1": corr(y[:-1], x[1:]) if n > 1 else 0.0,
        "sp_lead_rate_next5_sum_corr": corr(x[:lead_n], rate_next5),
        "rate_lead_sp_next5_sum_corr": corr(y[:lead_n], sp_next5),
        "sp_vol_lead_rate_vol_corr_1": corr(sp_vol20[:-1], rate_vol20[1:]) if vol_n > 1 else 0.0,
        "rate_vol_lead_sp_vol_corr_1": corr(rate_vol20[:-1], sp_vol20[1:]) if vol_n > 1 else 0.0,
    }


def volatility_downside_features(
    sp: list[float], rate: list[float], window: int = 60
) -> dict[str, float]:
    rolling_correlations = rolling_corr(sp, rate, window)
    sp_volatility = rolling_std(sp, window)
    rate_volatility = rolling_std(rate, window)
    combined_volatility = [
        math.sqrt(max(sp_value * rate_value, 0.0))
        for sp_value, rate_value in zip(sp_volatility, rate_volatility)
    ]
    absolute_correlations = [abs(value) for value in rolling_correlations]
    high_volatility_threshold = quantile(combined_volatility, 0.80)
    high_volatility_correlations = [
        correlation
        for volatility, correlation in zip(
            combined_volatility, absolute_correlations
        )
        if volatility >= high_volatility_threshold
    ]
    normal_volatility_correlations = [
        correlation
        for volatility, correlation in zip(
            combined_volatility, absolute_correlations
        )
        if volatility < high_volatility_threshold
    ]

    sp_q10 = quantile(sp, 0.10)
    sp_q20 = quantile(sp, 0.20)
    sp_q80 = quantile(sp, 0.80)
    downside_pairs = [
        (sp_value, rate_value)
        for sp_value, rate_value in zip(sp, rate)
        if sp_value <= sp_q10
    ]
    central_pairs = [
        (sp_value, rate_value)
        for sp_value, rate_value in zip(sp, rate)
        if sp_q20 <= sp_value <= sp_q80
    ]
    downside_corr = corr(
        [pair[0] for pair in downside_pairs],
        [pair[1] for pair in downside_pairs],
    )
    central_corr = corr(
        [pair[0] for pair in central_pairs],
        [pair[1] for pair in central_pairs],
    )
    return {
        "vol_abs_corr_coupling_60": corr(
            combined_volatility, absolute_correlations
        ),
        "high_vol_abs_corr_gap_60": mean(high_volatility_correlations)
        - mean(normal_volatility_correlations),
        "sp_downside_corr_gap_10": downside_corr - central_corr,
        "sp_downside_abs_corr_gap_10": abs(downside_corr) - abs(central_corr),
    }


def joint_percentile_features(
    sp_abs: list[float], rate_abs: list[float]
) -> dict[str, float]:
    if not sp_abs or not rate_abs:
        return {
            **{name: 0.0 for name in JOINT_DIAGNOSTIC_COLUMNS},
            JOINT_EQUAL_MEAN_FEATURE: 0.0,
        }
    rates: dict[str, float] = {}
    n = min(len(sp_abs), len(rate_abs))
    for percentile in range(90, 100):
        q = percentile / 100.0
        sp_threshold = quantile(sp_abs, q)
        rate_threshold = quantile(rate_abs, q)
        rates[f"joint_large_move_{percentile}"] = sum(
            a > sp_threshold and b > rate_threshold
            for a, b in zip(sp_abs[:n], rate_abs[:n])
        ) / n
    rates[JOINT_EQUAL_MEAN_FEATURE] = mean(
        [rates[name] for name in JOINT_DIAGNOSTIC_COLUMNS]
    )
    return rates


def joint_directional_95_features(
    sp: list[float], rate: list[float], sp_q95: float, rate_q95: float
) -> dict[str, float]:
    n = min(len(sp), len(rate))
    counts = {name: 0 for name in JOINT_DIRECTIONAL_FEATURES}
    if n == 0:
        return {name: 0.0 for name in JOINT_DIRECTIONAL_FEATURES}
    for sp_value, rate_value in zip(sp[:n], rate[:n]):
        if abs(sp_value) <= sp_q95 or abs(rate_value) <= rate_q95:
            continue
        sp_sign = "pos" if sp_value > 0.0 else "neg"
        rate_sign = "pos" if rate_value > 0.0 else "neg"
        counts[f"joint_large_move_95_sp_{sp_sign}_rate_{rate_sign}"] += 1
    return {name: count / n for name, count in counts.items()}


def empirical_percentile_ranks(xs: list[float]) -> list[float]:
    """Average-tie empirical ranks scaled to [0, 1]."""
    n = len(xs)
    if n < 2:
        return [0.5] * n
    positions: dict[float, list[int]] = {}
    for index, value in enumerate(xs):
        positions.setdefault(value, []).append(index)
    ranks = [0.0] * n
    seen = 0
    for value in sorted(positions):
        indices = positions[value]
        average_rank = seen + (len(indices) - 1) / 2.0
        scaled_rank = average_rank / (n - 1)
        for index in indices:
            ranks[index] = scaled_rank
        seen += len(indices)
    return ranks


def smooth_joint_rank_feature(
    sp_abs: list[float], rate_abs: list[float], center: float = 0.95,
    tau: float = 0.02,
) -> float:
    """Soft co-exceedance near the 95th empirical percentile."""
    n = min(len(sp_abs), len(rate_abs))
    if n == 0:
        return 0.0
    sp_ranks = empirical_percentile_ranks(sp_abs[:n])
    rate_ranks = empirical_percentile_ranks(rate_abs[:n])

    def soft_tail_weight(rank: float) -> float:
        z = max(min((rank - center) / tau, 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-z))

    return mean([
        soft_tail_weight(sp_rank) * soft_tail_weight(rate_rank)
        for sp_rank, rate_rank in zip(sp_ranks, rate_ranks)
    ])


def features_for_pair(
    sp: list[float],
    rate: list[float],
    rolling_corr_window: int | None = None,
    joint_feature_mode: str = "single_95",
    dependence_feature_mode: str = "none",
) -> dict[str, float]:
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
    joint_features = joint_percentile_features(sp_abs, rate_abs)
    smooth_joint = smooth_joint_rank_feature(sp_abs, rate_abs)
    features = {
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
        "sp_abs_future_vol_corr_5": abs_future_vol_corr(sp, 5),
        "sp_leverage_corr_1": leverage_corr(sp, 1),
        "sp_leverage_corr_5": leverage_corr(sp, 5),
        "rate_acf1": autocorr(rate, 1),
        "rate_acf5": autocorr(rate, 5),
        "rate_abs_acf1": autocorr(rate_abs, 1),
        "rate_abs_acf5": autocorr(rate_abs, 5),
        "rate_abs_future_vol_corr_5": abs_future_vol_corr(rate, 5),
        "corr_sp_rate": corr(sp, rate),
        "joint_large_move_95": joint_features["joint_large_move_95"],
        "sp_tail_ratio": sp_q99 / sp_std if sp_std != 0.0 else 0.0,
        "rate_tail_ratio": rate_q99 / rate_std if rate_std != 0.0 else 0.0,
        **cross_asset_features(sp, rate),
    }
    if joint_feature_mode == "equal_mean_90_99":
        features.pop("joint_large_move_95")
        features.update(joint_features)
    elif joint_feature_mode == "directional_95":
        features.pop("joint_large_move_95")
        features.update(
            joint_directional_95_features(sp, rate, sp_q95, rate_q95)
        )
    elif joint_feature_mode == "none":
        features.pop("joint_large_move_95")
    elif joint_feature_mode == "smooth_rank_95_tau02":
        features.pop("joint_large_move_95")
        features[JOINT_SMOOTH_RANK_FEATURE] = smooth_joint
    elif joint_feature_mode != "single_95":
        raise ValueError(f"Unknown joint_feature_mode: {joint_feature_mode}")
    if rolling_corr_window is not None:
        values = rolling_corr(sp, rate, rolling_corr_window)
        prefix = f"rolling_corr_{rolling_corr_window}"
        features.update(
            {
                f"{prefix}_mean": mean(values),
                f"{prefix}_std": std(values),
                f"{prefix}_q10": quantile(values, 0.10),
                f"{prefix}_q50": quantile(values, 0.50),
                f"{prefix}_q90": quantile(values, 0.90),
                f"{prefix}_min": min(values) if values else 0.0,
                f"{prefix}_max": max(values) if values else 0.0,
                f"{prefix}_negative_fraction": (
                    sum(value < 0.0 for value in values) / len(values) if values else 0.0
                ),
            }
        )
    if dependence_feature_mode == "vol_downside_60_q10":
        features.update(volatility_downside_features(sp, rate))
    elif dependence_feature_mode != "none":
        raise ValueError(
            f"Unknown dependence_feature_mode: {dependence_feature_mode}"
        )
    return features


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
    data_name = config.get("data_name", "mf2_garch_regime_corr")
    SPLIT_DIR = experiment_dir / "data" / "splits" / data_name
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
