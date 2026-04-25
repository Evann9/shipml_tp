from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import LabelEncoder


if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parent))
    from ship_type_model import (  # type: ignore  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        refit_full,
        save_json,
        split_columns,
    )
else:
    from .ship_type_model import (  # noqa: E402
        RANDOM_STATE,
        TARGET,
        load_type_data,
        metrics_summary,
        model_specs,
        refit_full,
        save_json,
        split_columns,
    )


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "ais_ship_type_with_mmsi.csv"
DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_group_split_evaluation.json"
)
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_group_split.joblib"
)
DEFAULT_MODEL_METRICS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_group_split_metrics.json"
)
DEFAULT_CLASS_METRICS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_class_metrics.csv"
)
DEFAULT_CONFUSION_PAIRS_PATH = (
    Path(__file__).resolve().parent / "outputs" / "ship_type_classifier_confusion_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate ship-type classifiers with a leakage-resistant group split. "
            "The grouping column, usually MMSI, is used only for splitting and is "
            "removed from model features."
        )
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="CSV containing MMSI and shiptype columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSON file where evaluation metrics are saved.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Saved deployable group-split-selected model bundle.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=DEFAULT_MODEL_METRICS_PATH,
        help="Compact metrics JSON for the deployable group-split-selected model.",
    )
    parser.add_argument(
        "--class-metrics-out",
        type=Path,
        default=DEFAULT_CLASS_METRICS_PATH,
        help="CSV with per-class precision/recall/F1/support for the best group-split model.",
    )
    parser.add_argument(
        "--confusion-pairs-out",
        type=Path,
        default=DEFAULT_CONFUSION_PAIRS_PATH,
        help="CSV with actual -> predicted confusion pairs for the best group-split model.",
    )
    parser.add_argument(
        "--group-col",
        type=str,
        default="mmsi",
        help="Group column used to prevent the same vessel from appearing in both splits.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Approximate test ratio. StratifiedGroupKFold uses 1/test_size folds.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["logistic_regression", "random_forest", "voting", "xgboost"],
        help=(
            "Models to compare. Supported: logistic_regression random_forest "
            "voting xgboost knn svc."
        ),
    )
    parser.add_argument(
        "--compare-random-split",
        action="store_true",
        help="Also run the old row-level random split to show the leakage gap.",
    )
    return parser.parse_args()


def resolve_column(df: pd.DataFrame, requested: str) -> str:
    lookup = {col.lower(): col for col in df.columns}
    resolved = lookup.get(requested.lower())
    if resolved is None:
        raise ValueError(
            f"Group column '{requested}' not found. Available columns: {list(df.columns)}"
        )
    return resolved


def add_trig_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for angle_col in ["cog", "heading"]:
        if angle_col not in df.columns:
            continue
        angle = pd.to_numeric(df[angle_col], errors="coerce")
        radians = np.radians(angle.fillna(0.0))
        sin_col = f"{angle_col}_sin"
        cos_col = f"{angle_col}_cos"
        if sin_col not in df.columns:
            df[sin_col] = np.sin(radians)
        if cos_col not in df.columns:
            df[cos_col] = np.cos(radians)
    return df


def group_train_test_split(
    x: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, str]:
    test_size = min(max(test_size, 0.05), 0.5)
    n_splits = max(2, int(round(1.0 / test_size)))
    group_counts = groups.value_counts()
    usable = len(group_counts) >= n_splits and y.nunique() > 1

    if usable:
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        method = f"StratifiedGroupKFold(n_splits={n_splits})"
    else:
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(x, y, groups))
        method = "GroupShuffleSplit"

    return (
        x.iloc[train_idx],
        x.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
        groups.iloc[train_idx],
        groups.iloc[test_idx],
        method,
    )


def evaluate_specs(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    model_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    categorical_cols, numeric_cols = split_columns(x_train)
    specs, skipped = model_specs(categorical_cols, numeric_cols, set(model_names))
    results: list[dict[str, Any]] = []

    for spec in specs:
        metrics = fit_spec_with_predictions(spec, x_train, x_test, y_train, y_test)
        results.append(metrics)

    results.sort(key=lambda item: (item["test_accuracy"], item["macro_f1"]), reverse=True)
    return results, skipped


def fit_spec_with_predictions(
    spec: Any,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    label_encoder: LabelEncoder | None = None
    fit_y: pd.Series | np.ndarray = y_train

    if spec.requires_label_encoding:
        label_encoder = LabelEncoder()
        fit_y = label_encoder.fit_transform(y_train)

    spec.estimator.fit(x_train, fit_y)
    pred = spec.estimator.predict(x_test)
    if label_encoder is not None:
        pred = label_encoder.inverse_transform(pred.astype(int))

    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    confusion_pairs = confusion_pair_rows(y_test, pred, spec.name, spec.display_name)
    return {
        "model_name": spec.name,
        "display_name": spec.display_name,
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_test, pred, average="weighted", zero_division=0)),
        "classification_report": report,
        "confusion_pairs": confusion_pairs,
        "top_confusion_pairs": confusion_pairs[:12],
    }


def confusion_pair_rows(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
    display_name: str,
) -> list[dict[str, Any]]:
    labels = sorted(set(y_true.astype(str)).union(set(map(str, y_pred))))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    rows: list[dict[str, Any]] = []
    for row_idx, actual in enumerate(labels):
        actual_total = int(cm[row_idx, :].sum())
        for col_idx, predicted in enumerate(labels):
            count = int(cm[row_idx, col_idx])
            if actual == predicted or count == 0:
                continue
            rows.append(
                {
                    "model_name": model_name,
                    "display_name": display_name,
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                    "actual_support": actual_total,
                    "actual_error_rate": float(count / max(actual_total, 1)),
                }
            )
    return sorted(rows, key=lambda item: item["count"], reverse=True)


def class_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    report = metrics.get("classification_report", {})
    rows: list[dict[str, Any]] = []
    for label, values in report.items():
        if not isinstance(values, dict) or label in {"accuracy", "macro avg", "weighted avg"}:
            continue
        rows.append(
            {
                "model_name": metrics["model_name"],
                "display_name": metrics["display_name"],
                "shiptype": label,
                "precision": values.get("precision"),
                "recall": values.get("recall"),
                "f1_score": values.get("f1-score"),
                "support": values.get("support"),
            }
        )
    return rows


def random_split_baseline(
    x: pd.DataFrame,
    y: pd.Series,
    model_names: list[str],
    test_size: float,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    return evaluate_specs(x_train, x_test, y_train, y_test, model_names)


def leakage_report(
    train_groups: pd.Series,
    test_groups: pd.Series,
) -> dict[str, Any]:
    train_set = set(train_groups.astype(str))
    test_set = set(test_groups.astype(str))
    overlap = train_set.intersection(test_set)
    return {
        "train_groups": int(len(train_set)),
        "test_groups": int(len(test_set)),
        "overlap_groups": int(len(overlap)),
        "overlap_ratio_of_test_groups": float(len(overlap) / max(len(test_set), 1)),
    }


def class_counts(y: pd.Series) -> dict[str, int]:
    return {str(label): int(count) for label, count in y.value_counts().sort_index().items()}


def train_deploy_bundle(
    x: pd.DataFrame,
    y: pd.Series,
    best_metrics: dict[str, Any],
    all_metrics: list[dict[str, Any]],
    skipped: dict[str, str],
    split_info: dict[str, Any],
) -> dict[str, Any]:
    categorical_cols, numeric_cols = split_columns(x)
    fresh_specs, _ = model_specs(categorical_cols, numeric_cols, {best_metrics["model_name"]})
    if not fresh_specs:
        raise RuntimeError(f"Could not recreate model spec: {best_metrics['model_name']}")

    final_estimator, final_label_encoder = refit_full(fresh_specs[0], x, y)
    return {
        "target": TARGET,
        "model_name": best_metrics["model_name"],
        "display_name": best_metrics["display_name"],
        "estimator": final_estimator,
        "label_encoder": final_label_encoder,
        "feature_columns": x.columns.tolist(),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "train_rows": int(len(x)),
        "target_classes": sorted(y.unique().tolist()),
        "best_metrics": best_metrics,
        "all_metrics": all_metrics,
        "skipped_models": skipped,
        "evaluation": {
            "method": "mmsi_group_split",
            "split": split_info,
            "note": (
                "The selected model is refit on all rows for deployment after "
                "MMSI group-split evaluation. MMSI is not used as a feature."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    df = add_trig_features(load_type_data(args.data.resolve()))
    group_col = resolve_column(df, args.group_col)
    df[group_col] = df[group_col].astype(str)

    feature_drop_cols = [TARGET, group_col]
    x = df.drop(columns=feature_drop_cols)
    y = df[TARGET].astype(str)
    groups = df[group_col]

    (
        x_train,
        x_test,
        y_train,
        y_test,
        train_groups,
        test_groups,
        split_method,
    ) = group_train_test_split(x, y, groups, args.test_size)

    group_results, skipped = evaluate_specs(x_train, x_test, y_train, y_test, args.models)

    split_info = {
        "method": split_method,
        "requested_test_size": args.test_size,
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "train_class_counts": class_counts(y_train),
        "test_class_counts": class_counts(y_test),
        "leakage_check": leakage_report(train_groups, test_groups),
    }

    output: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(args.data.resolve()),
        "target": TARGET,
        "group_col": group_col,
        "group_col_used_as_feature": False,
        "feature_columns": x.columns.tolist(),
        "split": split_info,
        "group_split_metrics": group_results,
        "skipped_models": skipped,
    }

    if args.compare_random_split:
        random_results, random_skipped = random_split_baseline(x, y, args.models, args.test_size)
        output["row_level_random_split_metrics"] = random_results
        output["row_level_random_split_note"] = (
            "This is the old row-level split. It may be optimistic because points "
            "from the same MMSI can appear in both train and test."
        )
        output["skipped_models"].update(random_skipped)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output.resolve(), output)

    if not group_results:
        raise RuntimeError("No group-split model results were produced.")

    best_group_result = group_results[0]
    class_rows = class_metric_rows(best_group_result)
    confusion_rows = best_group_result.get("confusion_pairs", [])
    pd.DataFrame(class_rows).to_csv(
        args.class_metrics_out.resolve(),
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(confusion_rows).to_csv(
        args.confusion_pairs_out.resolve(),
        index=False,
        encoding="utf-8-sig",
    )

    bundle = train_deploy_bundle(
        x=x,
        y=y,
        best_metrics=best_group_result,
        all_metrics=group_results,
        skipped=skipped,
        split_info=split_info,
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out.resolve(), compress=3)
    save_json(args.metrics_out.resolve(), metrics_summary(bundle))

    print(f"Saved group-split metrics: {args.output.resolve()}")
    print(f"Saved deploy model: {args.model_out.resolve()}")
    print(f"Saved deploy model metrics: {args.metrics_out.resolve()}")
    print(f"Saved class metrics: {args.class_metrics_out.resolve()}")
    print(f"Saved confusion pairs: {args.confusion_pairs_out.resolve()}")
    print(f"Rows: train={len(x_train):,}, test={len(x_test):,}")
    print(
        "Group leakage check: "
        f"overlap={output['split']['leakage_check']['overlap_groups']:,} groups"
    )
    print("\nGroup split results:")
    for result in group_results:
        print(
            f"- {result['display_name']}: "
            f"accuracy={result['test_accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, "
            f"weighted_f1={result['weighted_f1']:.4f}"
        )

    if args.compare_random_split:
        print("\nRow-level random split results:")
        for result in output["row_level_random_split_metrics"]:
            print(
                f"- {result['display_name']}: "
                f"accuracy={result['test_accuracy']:.4f}, "
                f"macro_f1={result['macro_f1']:.4f}, "
                f"weighted_f1={result['weighted_f1']:.4f}"
            )


if __name__ == "__main__":
    main()
