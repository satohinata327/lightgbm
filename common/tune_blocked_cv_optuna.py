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


def suggested_params(trial: optuna.Trial, fixed: dict[str, object]) -> dict[str, object]:
    max_depth = trial.suggest_int("max_depth", 2, 6)
    proposed_leaves = trial.suggest_int("num_leaves", 3, 31)
    params = dict(fixed)
    params.update(
        {
            "max_depth": max_depth,
            "num_leaves": min(proposed_leaves, 2**max_depth - 1),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 120, step=10),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0, step=0.1),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
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
        x_train, y_train = xy(train_rows, np)
        x_val, y_val = xy(val_rows, np)
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
        fold_results.append(
            {
                "fold": fold_no,
                "train_blocks": train_ids,
                "validation_block": validation_block,
                "best_iteration": int(model.best_iteration_),
                "metrics": metrics_for(y_val, prob, pred, metric_functions),
            }
        )
    return fold_results


def aggregate(folds: list[dict[str, object]], penalty: float) -> dict[str, float]:
    aucs = [float(fold["metrics"]["auc"]) for fold in folds]
    recalls = [float(fold["metrics"]["real_recall"]) for fold in folds]
    balanced = [float(fold["metrics"]["balanced_accuracy"]) for fold in folds]
    loglosses = [float(fold["metrics"]["logloss"]) for fold in folds]
    auc_std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
    return {
        "robust_auc": statistics.mean(aucs) - penalty * auc_std,
        "auc_mean": statistics.mean(aucs),
        "auc_std": auc_std,
        "auc_min": min(aucs),
        "real_recall_mean": statistics.mean(recalls),
        "balanced_accuracy_mean": statistics.mean(balanced),
        "logloss_mean": statistics.mean(loglosses),
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
    penalty = float(config["optuna"]["auc_std_penalty"])

    def objective(trial: optuna.Trial) -> float:
        params = suggested_params(trial, fixed)
        folds = evaluate_folds(
            params, real_blocks, generated_blocks, validation_blocks, stopping, libs
        )
        summary = aggregate(folds, penalty)
        for name, value in summary.items():
            trial.set_user_attr(name, value)
        trial.set_user_attr("effective_num_leaves", params["num_leaves"])
        trial.set_user_attr(
            "best_iterations", [int(fold["best_iteration"]) for fold in folds]
        )
        return summary["robust_auc"]

    db_path = (tuning_dir / "study.db").resolve()
    study = optuna.create_study(
        study_name=config["experiment_name"],
        storage=f"sqlite:///{db_path}",
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=int(config["optuna"]["sampler_seed"])),
    )
    remaining = max(0, int(config["optuna"]["n_trials"]) - len(study.trials))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=remaining, gc_after_trial=True, show_progress_bar=True)
    write_trials(tuning_dir / "trials.csv", study)

    best_trial = study.best_trial
    best_params = suggested_params(
        optuna.trial.FixedTrial(best_trial.params), fixed
    )
    best_folds = evaluate_folds(
        best_params, real_blocks, generated_blocks, validation_blocks, stopping, libs
    )
    best_aggregate = aggregate(best_folds, penalty)
    best_iterations = [int(fold["best_iteration"]) for fold in best_folds]
    final_n_estimators = max(1, int(statistics.median(best_iterations)))
    final_params = dict(best_params)
    final_params["n_estimators"] = final_n_estimators

    final_train_ids = list(range(final_test_block))
    train_rows = [row for idx in final_train_ids for row in real_blocks[idx] + generated_blocks[idx]]
    test_rows = real_blocks[final_test_block] + generated_blocks[final_test_block]
    x_train, y_train = xy(train_rows, np)
    x_test, y_test = xy(test_rows, np)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    model = lgb.LGBMClassifier(**final_params)
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = model.predict(x_test)
    test_metrics = metrics_for(y_test, prob, pred, metric_functions)
    with (model_dir / "lgb_model.pkl").open("wb") as f:
        pickle.dump(model, f)
    with (model_dir / "scaler.pkl").open("wb") as f:
        pickle.dump(scaler, f)
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
