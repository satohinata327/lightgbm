#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
from pathlib import Path

import optuna


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_features import FEATURE_COLUMNS  # noqa: E402
from train_blocked_cv import metrics_for, packages, write_predictions, xy  # noqa: E402


def read_rows(path: Path) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_blocks(source_dir: Path, n_blocks: int):
    real = [read_rows(source_dir / f"real_block_{idx + 1}.csv") for idx in range(n_blocks)]
    generated = [
        read_rows(source_dir / f"generated_block_{idx + 1}.csv") for idx in range(n_blocks)
    ]
    return real, generated


def suggested_params(
    trial: optuna.Trial,
    fixed: dict[str, object],
    search: dict[str, object] | None = None,
) -> dict[str, object]:
    search = search or {}
    max_depth_range = search.get("max_depth", [2, 6])
    num_leaves_range = search.get("num_leaves", [3, 31])
    min_child_range = search.get("min_child_samples", [20, 120, 10])
    colsample_range = search.get("colsample_bytree", [0.4, 1.0, 0.1])
    subsample_range = search.get("subsample", [0.6, 1.0, 0.1])
    subsample_freq_range = search.get("subsample_freq", [1, 7])
    reg_alpha_range = search.get("reg_alpha", [1e-8, 10.0])
    reg_lambda_range = search.get("reg_lambda", [1e-8, 50.0])
    min_split_gain_range = search.get("min_split_gain", [0.0, 0.2])
    max_depth = trial.suggest_int("max_depth", *max_depth_range)
    proposed_leaves = trial.suggest_int("num_leaves", *num_leaves_range)
    params = dict(fixed)
    if "learning_rate" in search:
        params["learning_rate"] = trial.suggest_float(
            "learning_rate", *search["learning_rate"], log=True
        )
    params.update(
        {
            "max_depth": max_depth,
            "num_leaves": min(proposed_leaves, 2**max_depth - 1),
            "min_child_samples": trial.suggest_int(
                "min_child_samples", min_child_range[0], min_child_range[1],
                step=min_child_range[2]
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", colsample_range[0], colsample_range[1],
                step=colsample_range[2]
            ),
            "subsample": trial.suggest_float(
                "subsample", subsample_range[0], subsample_range[1],
                step=subsample_range[2]
            ),
            "subsample_freq": trial.suggest_int("subsample_freq", *subsample_freq_range),
            "reg_alpha": (
                0.0
                if search.get("allow_zero_regularization")
                and trial.suggest_categorical("reg_alpha_is_zero", [True, False])
                else trial.suggest_float("reg_alpha", *reg_alpha_range, log=True)
            ),
            "reg_lambda": (
                0.0
                if search.get("allow_zero_regularization")
                and trial.suggest_categorical("reg_lambda_is_zero", [True, False])
                else trial.suggest_float("reg_lambda", *reg_lambda_range, log=True)
            ),
            "min_split_gain": trial.suggest_float("min_split_gain", *min_split_gain_range),
        }
    )
    return params


def evaluate_folds(params, real_blocks, generated_blocks, validation_blocks, stopping, libs):
    lgb, np, StandardScaler, metric_functions = libs
    fold_results = []
    for fold_no, validation_block in enumerate(validation_blocks, start=1):
        train_ids = list(range(validation_block))
        train_rows = [row for idx in train_ids for row in real_blocks[idx] + generated_blocks[idx]]
        val_rows = real_blocks[validation_block] + generated_blocks[validation_block]
        x_train, y_train = xy(train_rows, np, FEATURE_COLUMNS)
        x_val, y_val = xy(val_rows, np, FEATURE_COLUMNS)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_val = scaler.transform(x_val)
        model = lgb.LGBMClassifier(**params)
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            eval_metric=stopping["eval_metric"],
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=int(stopping["stopping_rounds"]), verbose=False
                )
            ],
        )
        prob = model.predict_proba(x_val)[:, 1]
        pred = model.predict(x_val)
        eps = 1e-15
        clipped = np.clip(prob, eps, 1.0 - eps)
        real_mask = y_val == 1
        generated_mask = y_val == 0
        real_logloss = float(-np.mean(np.log(clipped[real_mask])))
        generated_logloss = float(-np.mean(np.log(1.0 - clipped[generated_mask])))
        balanced_logloss = 0.5 * (real_logloss + generated_logloss)
        fold_metrics = metrics_for(y_val, prob, pred, metric_functions)
        fold_metrics.update(
            {
                "real_logloss": real_logloss,
                "generated_logloss": generated_logloss,
                "balanced_logloss": balanced_logloss,
            }
        )
        fold_results.append(
            {
                "fold": fold_no,
                "train_blocks": train_ids,
                "validation_block": validation_block,
                "best_iteration": int(model.best_iteration_),
                "metrics": fold_metrics,
            }
        )
    return fold_results


def aggregate(
    folds: list[dict[str, object]], auc_std_penalty: float, worst_fold_penalty: float = 0.5
) -> dict[str, float]:
    aucs = [float(fold["metrics"]["auc"]) for fold in folds]
    recalls = [float(fold["metrics"]["real_recall"]) for fold in folds]
    balanced = [float(fold["metrics"]["balanced_accuracy"]) for fold in folds]
    loglosses = [float(fold["metrics"]["logloss"]) for fold in folds]
    balanced_loglosses = [
        float(fold["metrics"]["balanced_logloss"]) for fold in folds
    ]
    auc_std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    return {
        "robust_auc": statistics.mean(aucs) - auc_std_penalty * auc_std,
        "auc_mean": statistics.mean(aucs),
        "auc_std": auc_std,
        "auc_min": min(aucs),
        "real_recall_mean": statistics.mean(recalls),
        "balanced_accuracy_mean": statistics.mean(balanced),
        "logloss_mean": statistics.mean(loglosses),
        "balanced_logloss_mean": statistics.mean(balanced_loglosses),
        "balanced_logloss_worst": max(balanced_loglosses),
        "balanced_logloss_objective": (
            statistics.mean(balanced_loglosses)
            + worst_fold_penalty * max(balanced_loglosses)
        ),
    }


def write_trials(path: Path, study: optuna.Study) -> None:
    param_names = sorted({name for trial in study.trials for name in trial.params})
    attr_names = sorted({name for trial in study.trials for name in trial.user_attrs})
    with path.open("w", newline="", encoding="utf-8") as f:
        fields = ["number", "value", "state", *[f"param_{x}" for x in param_names], *attr_names]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for trial in study.trials:
            row = {"number": trial.number, "value": trial.value, "state": trial.state.name}
            row.update({f"param_{name}": trial.params.get(name, "") for name in param_names})
            row.update({name: trial.user_attrs.get(name, "") for name in attr_names})
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    experiment_dir = ROOT / config["experiment_dir"]
    tuning_dir = experiment_dir / "tuning"
    model_dir = experiment_dir / "model"
    results_dir = experiment_dir / "results"
    for path in [tuning_dir, model_dir, results_dir]:
        path.mkdir(parents=True, exist_ok=True)

    source_dir = ROOT / config["source_blocked_features"]
    blocked = config["blocked_cv"]
    n_blocks = int(blocked["n_blocks"])
    validation_blocks = [int(x) for x in blocked["validation_blocks"]]
    final_test_block = int(blocked["final_test_block"])
    real_blocks, generated_blocks = load_blocks(source_dir, n_blocks)
    libs = packages()
    lgb, np, StandardScaler, metric_functions = libs
    fixed = dict(config["fixed_lightgbm_params"])
    fixed.setdefault("verbosity", -1)
    stopping = config["early_stopping"]
    objective_name = config["optuna"].get("objective", "robust_auc")
    auc_std_penalty = float(config["optuna"].get("auc_std_penalty", 0.25))
    worst_fold_penalty = float(config["optuna"].get("worst_fold_penalty", 0.5))
    search_space = config["optuna"].get("search_space", {})

    def objective(trial: optuna.Trial) -> float:
        params = suggested_params(trial, fixed, search_space)
        folds = evaluate_folds(
            params, real_blocks, generated_blocks, validation_blocks, stopping, libs
        )
        summary = aggregate(folds, auc_std_penalty, worst_fold_penalty)
        for name, value in summary.items():
            trial.set_user_attr(name, value)
        for fold in folds:
            fold_no = int(fold["fold"])
            trial.set_user_attr(f"fold_{fold_no}_auc", fold["metrics"]["auc"])
            trial.set_user_attr(f"fold_{fold_no}_logloss", fold["metrics"]["logloss"])
            trial.set_user_attr(
                f"fold_{fold_no}_balanced_logloss",
                fold["metrics"]["balanced_logloss"],
            )
            trial.set_user_attr(
                f"fold_{fold_no}_real_logloss", fold["metrics"]["real_logloss"]
            )
            trial.set_user_attr(
                f"fold_{fold_no}_generated_logloss",
                fold["metrics"]["generated_logloss"],
            )
            trial.set_user_attr(
                f"fold_{fold_no}_real_recall", fold["metrics"]["real_recall"]
            )
        trial.set_user_attr("effective_num_leaves", params["num_leaves"])
        trial.set_user_attr(
            "best_iterations", [int(fold["best_iteration"]) for fold in folds]
        )
        return summary[objective_name]

    db_path = (tuning_dir / "study.db").resolve()
    direction = "minimize" if objective_name == "balanced_logloss_objective" else "maximize"
    study = optuna.create_study(
        study_name=config["experiment_name"],
        storage=f"sqlite:///{db_path}",
        direction=direction,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(config["optuna"]["sampler_seed"])),
    )
    if not study.trials and config["optuna"].get("baseline_trial"):
        study.enqueue_trial(config["optuna"]["baseline_trial"])
    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials
    )
    remaining = max(0, int(config["optuna"]["n_trials"]) - completed_trials)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=True)
    write_trials(tuning_dir / "trials.csv", study)

    best_trial = study.best_trial
    best_params = suggested_params(
        optuna.trial.FixedTrial(best_trial.params), fixed, search_space
    )
    best_folds = evaluate_folds(
        best_params, real_blocks, generated_blocks, validation_blocks, stopping, libs
    )
    best_aggregate = aggregate(best_folds, auc_std_penalty, worst_fold_penalty)
    best_iterations = [int(fold["best_iteration"]) for fold in best_folds]
    final_n_estimators = max(1, int(statistics.median(best_iterations)))
    final_params = dict(best_params)
    final_params["n_estimators"] = final_n_estimators

    final_train_ids = list(range(final_test_block))
    train_rows = [row for idx in final_train_ids for row in real_blocks[idx] + generated_blocks[idx]]
    test_rows = real_blocks[final_test_block] + generated_blocks[final_test_block]
    x_train, y_train = xy(train_rows, np, FEATURE_COLUMNS)
    x_test, y_test = xy(test_rows, np, FEATURE_COLUMNS)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = lgb.LGBMClassifier(**final_params)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    test_metrics = metrics_for(y_test, prob, pred, metric_functions)
    eps = 1e-15
    clipped = np.clip(prob, eps, 1.0 - eps)
    real_mask = y_test == 1
    generated_mask = y_test == 0
    test_real_logloss = float(-np.mean(np.log(clipped[real_mask])))
    test_generated_logloss = float(
        -np.mean(np.log(1.0 - clipped[generated_mask]))
    )
    test_metrics.update(
        {
            "real_logloss": test_real_logloss,
            "generated_logloss": test_generated_logloss,
            "balanced_logloss": 0.5
            * (test_real_logloss + test_generated_logloss),
        }
    )
    with (model_dir / "lgb_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (model_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
    (model_dir / "feature_config.json").write_text(
        json.dumps({"feature_columns": FEATURE_COLUMNS, "rolling_corr_window": None}, indent=2),
        encoding="utf-8",
    )
    write_predictions(results_dir / "test_predictions.csv", test_rows, y_test, prob, pred)

    importance = sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_), key=lambda x: x[1], reverse=True
    )
    with (results_dir / "feature_importance.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "importance"])
        writer.writerows(importance)

    best_output = {
        "best_trial": best_trial.number,
        "optuna_objective": objective_name,
        "objective_value": best_trial.value,
        "suggested_params": best_trial.params,
        "effective_params": best_params,
        "folds": best_folds,
        "cv": best_aggregate,
        "best_iterations": best_iterations,
        "final_n_estimators": final_n_estimators,
        "final_params": final_params,
        "final_test_metrics": test_metrics,
    }
    (tuning_dir / "best_params.json").write_text(
        json.dumps(best_output, indent=2), encoding="utf-8"
    )
    (results_dir / "metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8"
    )
    (results_dir / "tuning_summary.json").write_text(
        json.dumps(best_output, indent=2), encoding="utf-8"
    )
    print(json.dumps(best_output, indent=2))


if __name__ == "__main__":
    main()
