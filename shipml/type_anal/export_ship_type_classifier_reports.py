from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_MODEL_PATH = DEFAULT_OUTPUT_DIR / "ship_type_classifier_group_split.joblib"
DEFAULT_CLASS_METRICS_PATH = DEFAULT_OUTPUT_DIR / "ship_type_classifier_class_metrics.csv"
DEFAULT_CONFUSION_PAIRS_PATH = DEFAULT_OUTPUT_DIR / "ship_type_classifier_confusion_pairs.csv"
DEFAULT_IMPORTANCE_CSV = DEFAULT_OUTPUT_DIR / "ship_type_classifier_feature_importance.csv"
DEFAULT_IMPORTANCE_PNG = DEFAULT_OUTPUT_DIR / "ship_type_classifier_feature_importance.png"
DEFAULT_CONFUSION_MATRIX_CSV = DEFAULT_OUTPUT_DIR / "ship_type_classifier_confusion_matrix.csv"
DEFAULT_CONFUSION_MATRIX_PNG = DEFAULT_OUTPUT_DIR / "ship_type_classifier_confusion_matrix.png"
DEFAULT_TOP_CONFUSION_PNG = DEFAULT_OUTPUT_DIR / "ship_type_classifier_top_confusion_pairs.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export feature importance and confusion charts for the group-split ship-type model."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--class-metrics", type=Path, default=DEFAULT_CLASS_METRICS_PATH)
    parser.add_argument("--confusion-pairs", type=Path, default=DEFAULT_CONFUSION_PAIRS_PATH)
    parser.add_argument("--importance-csv", type=Path, default=DEFAULT_IMPORTANCE_CSV)
    parser.add_argument("--importance-png", type=Path, default=DEFAULT_IMPORTANCE_PNG)
    parser.add_argument("--matrix-csv", type=Path, default=DEFAULT_CONFUSION_MATRIX_CSV)
    parser.add_argument("--matrix-png", type=Path, default=DEFAULT_CONFUSION_MATRIX_PNG)
    parser.add_argument("--top-confusion-png", type=Path, default=DEFAULT_TOP_CONFUSION_PNG)
    parser.add_argument("--top-n", type=int, default=20)
    return parser.parse_args()


def load_model_bundle(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Model bundle not found: {path}")
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "estimator" not in bundle:
        raise ValueError(f"{path} is not a supported ship-type model bundle.")
    return bundle


def export_feature_importance(bundle: dict[str, Any], csv_path: Path, png_path: Path, top_n: int) -> None:
    estimator = bundle["estimator"]
    feature_columns = bundle.get("feature_columns", [])
    classifier = estimator.named_steps.get("classifier") if hasattr(estimator, "named_steps") else estimator

    if classifier is None:
        raise ValueError("Could not find classifier step in model pipeline.")

    importances = feature_importance_values(classifier)
    feature_names = transformed_feature_names(estimator, feature_columns)
    if len(feature_names) != len(importances):
        raise ValueError(
            "Feature name count does not match model importance count: "
            f"{len(feature_names)} names vs {len(importances)} values."
        )

    rows = []
    for raw_name, importance in zip(feature_names, importances):
        feature = clean_feature_name(raw_name)
        rows.append(
            {
                "feature": feature,
                "source_feature": source_feature(feature),
                "importance": float(importance),
            }
        )

    importance_df = (
        pd.DataFrame(rows)
        .sort_values("importance", ascending=False, ignore_index=True)
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    save_importance_plot(importance_df.head(max(top_n, 1)), png_path)


def feature_importance_values(classifier: Any) -> np.ndarray:
    if hasattr(classifier, "feature_importances_"):
        return np.asarray(classifier.feature_importances_, dtype=float)
    if hasattr(classifier, "coef_"):
        return np.mean(np.abs(np.asarray(classifier.coef_, dtype=float)), axis=0)
    raise ValueError(
        "The saved classifier does not expose feature_importances_ or coef_. "
        "Use a tree-based model or linear model for this export."
    )


def transformed_feature_names(estimator: Any, feature_columns: list[str]) -> list[str]:
    if hasattr(estimator, "named_steps") and "preprocessor" in estimator.named_steps:
        preprocessor = estimator.named_steps["preprocessor"]
        try:
            return [str(name) for name in preprocessor.get_feature_names_out(feature_columns)]
        except TypeError:
            return [str(name) for name in preprocessor.get_feature_names_out()]
    return [str(name) for name in feature_columns]


def clean_feature_name(name: str) -> str:
    cleaned = name
    for prefix in ["num__", "cat__"]:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
    if "_" in cleaned:
        for source in ["navigationalstatus"]:
            marker = f"{source}_"
            if cleaned.startswith(marker):
                return f"{source}={cleaned[len(marker):]}"
    return cleaned


def source_feature(feature: str) -> str:
    if "=" in feature:
        return feature.split("=", 1)[0]
    return feature


def export_confusion_outputs(
    class_metrics_path: Path,
    confusion_pairs_path: Path,
    matrix_csv_path: Path,
    matrix_png_path: Path,
    top_confusion_png_path: Path,
    top_n: int,
) -> None:
    class_metrics = pd.read_csv(class_metrics_path)
    confusion_pairs = pd.read_csv(confusion_pairs_path)

    labels = class_metrics["shiptype"].astype(str).tolist()
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)

    for row in confusion_pairs.itertuples(index=False):
        actual = str(row.actual)
        predicted = str(row.predicted)
        if actual in matrix.index and predicted in matrix.columns:
            matrix.loc[actual, predicted] = int(row.count)

    support_by_label = (
        class_metrics.set_index("shiptype")["support"]
        .fillna(0)
        .astype(int)
        .to_dict()
    )
    for label in labels:
        off_diagonal = int(matrix.loc[label].sum())
        matrix.loc[label, label] = max(int(support_by_label.get(label, 0)) - off_diagonal, 0)

    matrix_csv_path.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(matrix_csv_path, index_label="actual", encoding="utf-8-sig")
    save_confusion_matrix_plot(matrix, matrix_png_path)
    save_top_confusion_plot(confusion_pairs.head(max(top_n, 1)), top_confusion_png_path)


def save_importance_plot(df: pd.DataFrame, path: Path) -> None:
    plt = import_matplotlib()
    plot_df = df.iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.32)))
    ax.barh(plot_df["feature"], plot_df["importance"], color="#0f766e")
    ax.set_title("Ship Type Feature Importance")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrix_plot(matrix: pd.DataFrame, path: Path) -> None:
    plt = import_matplotlib()
    row_totals = matrix.sum(axis=1).replace(0, np.nan)
    normalized = matrix.div(row_totals, axis=0).fillna(0.0) * 100.0

    fig_size = max(7, len(matrix) * 0.62)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    image = ax.imshow(normalized.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    ax.set_title("Ship Type Confusion Matrix (% by actual class)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(matrix.index)), matrix.index)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_top_confusion_plot(df: pd.DataFrame, path: Path) -> None:
    plt = import_matplotlib()
    if df.empty:
        return
    plot_df = df.copy()
    plot_df["pair"] = plot_df["actual"].astype(str) + " -> " + plot_df["predicted"].astype(str)
    plot_df = plot_df.iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, max(4, len(plot_df) * 0.36)))
    ax.barh(plot_df["pair"], plot_df["count"], color="#be185d")
    ax.set_title("Top Ship Type Confusion Pairs")
    ax.set_xlabel("Count")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def import_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save PNG charts.") from exc
    return plt


def main() -> None:
    args = parse_args()
    bundle = load_model_bundle(args.model.resolve())
    export_feature_importance(
        bundle=bundle,
        csv_path=args.importance_csv.resolve(),
        png_path=args.importance_png.resolve(),
        top_n=args.top_n,
    )
    export_confusion_outputs(
        class_metrics_path=args.class_metrics.resolve(),
        confusion_pairs_path=args.confusion_pairs.resolve(),
        matrix_csv_path=args.matrix_csv.resolve(),
        matrix_png_path=args.matrix_png.resolve(),
        top_confusion_png_path=args.top_confusion_png.resolve(),
        top_n=args.top_n,
    )
    print(f"Saved feature importance CSV: {args.importance_csv.resolve()}")
    print(f"Saved feature importance chart: {args.importance_png.resolve()}")
    print(f"Saved confusion matrix CSV: {args.matrix_csv.resolve()}")
    print(f"Saved confusion matrix chart: {args.matrix_png.resolve()}")
    print(f"Saved top confusion chart: {args.top_confusion_png.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise
