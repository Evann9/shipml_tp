from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, train_test_split


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from train_ship_type_classifier_group_split import (  # type: ignore  # noqa: E402
        DEFAULT_DATA_PATH,
        add_trig_features,
        class_counts,
        group_train_test_split,
        leakage_report,
        resolve_column,
    )
    from ship_type_model import (  # type: ignore  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        save_json,
        split_columns,
    )
else:
    from .train_ship_type_classifier_group_split import (  # noqa: E402
        DEFAULT_DATA_PATH,
        add_trig_features,
        class_counts,
        group_train_test_split,
        leakage_report,
        resolve_column,
    )
    from .ship_type_model import (  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        save_json,
        split_columns,
    )


DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_tuning_results.json"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_tuned_group_split.joblib"
)
DEFAULT_MODEL_METRICS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_tuned_group_split_metrics.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune the MMSI group-split ship-type RandomForest model without "
            "letting the same vessel appear in both train and validation folds."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV containing mmsi and shiptype columns.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="mmsi",
        help="Group column used for leakage-safe splitting.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Detailed tuning/evaluation metrics JSON.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved tuned model bundle.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_MODEL_METRICS_PATH,
        help="Compact metrics JSON for the tuned model.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="External MMSI group holdout ratio.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=3,
        help="Internal StratifiedGroupKFold folds for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=12,
        help="Number of random hyperparameter combinations to test.",
    )
    parser.add_argument(
        "--max-search-groups",
        type=int,
        default=0,
        help=(
            "Optional cap on MMSI groups used only during the search. "
            "0 means use all training groups."
        ),
    )
    parser.add_argument(
        "--search-n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for RandomizedSearchCV.",
    )
    parser.add_argument(
        "--estimator-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs inside RandomForest.",
    )
    parser.add_argument(
        "--scoring",
        type=str,
        default="macro_f1",
        choices=["macro_f1", "weighted_f1", "accuracy"],
        help="Metric used to choose the best hyperparameters.",
    )
    return parser.parse_args()


def parameter_space() -> dict[str, list[Any]]:
    return {
        "classifier__n_estimators": [200, 300, 500, 700],
        "classifier__max_depth": [None, 12, 20, 32, 48],
        "classifier__min_samples_split": [2, 5, 10, 20],
        "classifier__min_samples_leaf": [1, 2, 4, 8],
        "classifier__max_features": ["sqrt", "log2", 0.5, None],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
        "classifier__bootstrap": [True, False],
    }


def scoring_map() -> dict[str, Any]:
    return {
        "macro_f1": "f1_macro",
        "weighted_f1": "f1_weighted",
        "accuracy": "accuracy",
    }


def prepare_data(
    data_path: Path,
    group_col_name: str,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, str]:
    df = add_trig_features(load_type_data(data_path))
    group_col = resolve_column(df, group_col_name)
    df[group_col] = df[group_col].astype(str)

    x = df.drop(columns=[TARGET, group_col])
    y = df[TARGET].astype(str)
    groups = df[group_col]
    return x, y, groups, group_col


def sample_search_groups(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    train_groups: pd.Series,
    max_groups: int,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    if max_groups <= 0 or train_groups.nunique() <= max_groups:
        return x_train, y_train, train_groups

    group_labels = (
        pd.DataFrame({"group": train_groups.astype(str), "target": y_train.astype(str)})
        .groupby("group", sort=False)["target"]
        .agg(lambda values: values.mode().iloc[0])
    )
    stratify = group_labels if group_labels.value_counts().min() >= 2 else None
    selected_groups, _ = train_test_split(
        group_labels.index.to_numpy(),
        train_size=max_groups,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )
    selected = set(map(str, selected_groups))
    mask = train_groups.astype(str).isin(selected)
    return x_train.loc[mask], y_train.loc[mask], train_groups.loc[mask]


def make_random_forest_pipeline(
    x_train: pd.DataFrame,
    estimator_n_jobs: int,
) -> Any:
    categorical_cols, numeric_cols = split_columns(x_train)
    specs, skipped = model_specs(categorical_cols, numeric_cols, {"random_forest"})
    if skipped:
        raise RuntimeError(f"RandomForest spec could not be built: {skipped}")
    if not specs:
        raise RuntimeError("RandomForest spec was not created.")
    estimator = specs[0].estimator
    estimator.set_params(classifier__n_jobs=estimator_n_jobs)
    return estimator


def top_confusion_pairs(
    y_true: pd.Series,
    y_pred: np.ndarray,
    top_n: int = 12,
) -> list[dict[str, Any]]:
    labels = sorted(set(y_true.astype(str)).union(set(map(str, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pairs: list[dict[str, Any]] = []
    for row_idx, actual in enumerate(labels):
        for col_idx, predicted in enumerate(labels):
            count = int(cm[row_idx, col_idx])
            if actual != predicted and count > 0:
                pairs.append(
                    {"actual": actual, "predicted": predicted, "count": count}
                )
    return sorted(pairs, key=lambda item: item["count"], reverse=True)[:top_n]


def evaluate_holdout(
    estimator: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[Any, dict[str, Any]]:
    model = clone(estimator)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "model_name": "random_forest_group_tuned",
        "display_name": "RandomForest Tuned (MMSI Group Split)",
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_test, pred, average="weighted", zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0,
        ),
        "top_confusion_pairs": top_confusion_pairs(y_test, pred),
    }
    return model, metrics


def compact_cv_results(search: RandomizedSearchCV, top_n: int = 10) -> list[dict[str, Any]]:
    results = pd.DataFrame(search.cv_results_)
    score_col = f"mean_test_{search.refit}" if isinstance(search.refit, str) else "mean_test_score"
    if score_col not in results.columns:
        score_col = "mean_test_score"

    cols = [
        "rank_test_" + search.refit if isinstance(search.refit, str) else "rank_test_score",
        score_col,
        "std_test_" + search.refit if isinstance(search.refit, str) else "std_test_score",
        "params",
    ]
    cols = [col for col in cols if col in results.columns]
    top = results.sort_values(score_col, ascending=False).head(top_n)
    return top[cols].to_dict("records")


def train_full_bundle(
    best_estimator: Any,
    x: pd.DataFrame,
    y: pd.Series,
    holdout_metrics: dict[str, Any],
    search: RandomizedSearchCV,
    split_info: dict[str, Any],
) -> dict[str, Any]:
    final_estimator = clone(best_estimator)
    final_estimator.fit(x, y)
    categorical_cols, numeric_cols = split_columns(x)
    best_metrics = {
        **holdout_metrics,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_refit_metric": str(search.refit),
    }
    return {
        "target": TARGET,
        "model_name": "random_forest_group_tuned",
        "display_name": "RandomForest Tuned (MMSI Group Split)",
        "estimator": final_estimator,
        "label_encoder": None,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(x)),
        "target_classes": sorted(y.unique().tolist()),
        "best_metrics": best_metrics,
        "all_metrics": [best_metrics],
        "skipped_models": {},
        "evaluation": {
            "method": "nested_mmsi_group_tuning",
            "split": split_info,
            "note": (
                "Hyperparameters were selected with StratifiedGroupKFold on "
                "the external training split, then evaluated on a separate "
                "MMSI group holdout. The saved estimator is refit on all rows."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    x, y, groups, group_col = prepare_data(args.data.resolve(), args.group_col)

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
        split_method,
    ) = group_train_test_split(x, y, groups, args.test_size)

    x_search, y_search, search_groups = sample_search_groups(
        x_train,
        y_train,
        train_groups,
        args.max_search_groups,
    )

    estimator = make_random_forest_pipeline(x_search, args.estimator_n_jobs)
    cv = StratifiedGroupKFold(
        n_splits=max(args.cv_folds, 2),
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    scorers = scoring_map()
    search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=parameter_space(),
        n_iter=max(args.n_iter, 1),
        scoring=scorers,
        refit=args.scoring,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=args.search_n_jobs,
        verbose=2,
        return_train_score=True,
        error_score="raise",
    )
    search.fit(x_search, y_search, groups=search_groups)

    tuned_estimator = clone(estimator).set_params(**search.best_params_)
    tuned_estimator.set_params(classifier__n_jobs=args.estimator_n_jobs)
    _, holdout_metrics = evaluate_holdout(
        tuned_estimator,
        x_train,
        x_test,
        y_train,
        y_test,
    )

    split_info = {
        "outer_split_method": split_method,
        "outer_requested_test_size": args.test_size,
        "outer_train_rows": int(len(x_train)),
        "outer_test_rows": int(len(x_test)),
        "outer_train_class_counts": class_counts(y_train),
        "outer_test_class_counts": class_counts(y_test),
        "outer_leakage_check": leakage_report(train_groups, test_groups),
        "inner_cv_method": f"StratifiedGroupKFold(n_splits={max(args.cv_folds, 2)})",
        "inner_search_rows": int(len(x_search)),
        "inner_search_groups": int(search_groups.nunique()),
        "group_col": group_col,
        "group_col_used_as_feature": False,
    }

    bundle = train_full_bundle(
        best_estimator=tuned_estimator,
        x=x,
        y=y,
        holdout_metrics=holdout_metrics,
        search=search,
        split_info=split_info,
    )

    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.resolve()),
        "target": TARGET,
        "group_col": group_col,
        "feature_columns": x.columns.tolist(),
        "scoring": args.scoring,
        "best_params": search.best_params_,
        "cv_best_score": float(search.best_score_),
        "cv_results_top": compact_cv_results(search),
        "holdout_metrics": holdout_metrics,
        "split": split_info,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.resolve(), output)
    joblib.dump(bundle, args.model_out.resolve(), compress=3)
    save_json(args.metrics_out.resolve(), metrics_summary(bundle))

    print(f"Saved tuning metrics: {args.output.resolve()}")
    print(f"Saved tuned model: {args.model_out.resolve()}")
    print(f"Saved tuned model metrics: {args.metrics_out.resolve()}")
    print(f"Best CV {args.scoring}: {search.best_score_:.4f}")
    print("Best params:")
    for key, value in search.best_params_.items():
        print(f"- {key}: {value}")
    print(
        "Holdout: "
        f"accuracy={holdout_metrics['test_accuracy']:.4f}, "
        f"macro_f1={holdout_metrics['macro_f1']:.4f}, "
        f"weighted_f1={holdout_metrics['weighted_f1']:.4f}"
    )
    print(
        "Group leakage check: "
        f"overlap={split_info['outer_leakage_check']['overlap_groups']:,} groups"
    )


if __name__ == "__main__":
    main()
