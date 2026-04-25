from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "outputs"
DEFAULT_DATA_PATH = ROOT_DIR / "ais_data_10day.csv"
DEFAULT_MODEL_PATH = OUTPUT_DIR / "future_position_regressor.joblib"
DEFAULT_METRICS_PATH = OUTPUT_DIR / "future_position_regressor_metrics.json"
DEFAULT_PREDICTIONS_PATH = OUTPUT_DIR / "future_position_forecast.csv"
RANDOM_STATE = 42
FEATURE_COLUMNS = [
    "Latitude",
    "Longitude",
    "SOG",
    "COG",
    "cog_sin",
    "cog_cos",
    "Width",
    "Length",
    "Draught",
    "delta_lat",
    "delta_lon",
    "delta_hours",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a leakage-aware future-position model and export web predictions."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-out", type=Path, default=DEFAULT_METRICS_PATH)
    parser.add_argument("--predictions-out", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tolerance-minutes", type=int, default=45)
    parser.add_argument("--n-estimators", type=int, default=80)
    parser.add_argument("--max-depth", type=int, default=18)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=180_000,
        help="Group-preserving cap for model fitting rows. Use 0 to fit all supervised rows.",
    )
    return parser.parse_args()


def load_ais_points(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"AIS data not found: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.astype(str).str.strip()
    rename = {
        "mmsi": "MMSI",
        "timestamp": "Timestamp",
        "latitude": "Latitude",
        "lat": "Latitude",
        "longitude": "Longitude",
        "lon": "Longitude",
        "lng": "Longitude",
        "sog": "SOG",
        "cog": "COG",
        "width": "Width",
        "length": "Length",
        "draught": "Draught",
    }
    df = df.rename(columns={key: value for key, value in rename.items() if key in df.columns})

    required = ["MMSI", "Timestamp", "Latitude", "Longitude"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {missing}")

    for col in ["SOG", "COG", "Width", "Length", "Draught"]:
        if col not in df.columns:
            df[col] = np.nan

    df["MMSI"] = df["MMSI"].astype(str)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    for col in ["Latitude", "Longitude", "SOG", "COG", "Width", "Length", "Draught"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["MMSI", "Timestamp", "Latitude", "Longitude"])
    df = df.loc[df["Latitude"].between(-90, 90) & df["Longitude"].between(-180, 180)]
    return df.sort_values(["MMSI", "Timestamp"], kind="mergesort").reset_index(drop=True)


def add_motion_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    radians = np.radians(pd.to_numeric(df["COG"], errors="coerce").fillna(0.0))
    df["cog_sin"] = np.sin(radians)
    df["cog_cos"] = np.cos(radians)

    grouped = df.groupby("MMSI", sort=False)
    df["prev_lat"] = grouped["Latitude"].shift(1)
    df["prev_lon"] = grouped["Longitude"].shift(1)
    df["prev_timestamp"] = grouped["Timestamp"].shift(1)
    df["delta_lat"] = (df["Latitude"] - df["prev_lat"]).fillna(0.0)
    df["delta_lon"] = (df["Longitude"] - df["prev_lon"]).fillna(0.0)
    df["delta_hours"] = (
        (df["Timestamp"] - df["prev_timestamp"]).dt.total_seconds() / 3600.0
    ).fillna(1.0)
    df["delta_hours"] = df["delta_hours"].clip(lower=0.05, upper=24.0)
    return df


def make_supervised_rows(
    df: pd.DataFrame,
    horizons: list[int],
    tolerance_minutes: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)

    for _, group in df.groupby("MMSI", sort=False):
        group = group.sort_values("Timestamp", kind="mergesort").reset_index(drop=True)
        if len(group) < max(horizons) + 1:
            continue

        merged = group.copy()
        for horizon in horizons:
            target = group[["Timestamp", "Latitude", "Longitude"]].copy()
            target = target.rename(
                columns={
                    "Timestamp": f"target_timestamp_{horizon}h",
                    "Latitude": f"target_lat_{horizon}h",
                    "Longitude": f"target_lon_{horizon}h",
                }
            )
            target["lookup_timestamp"] = (
                target[f"target_timestamp_{horizon}h"] - pd.Timedelta(hours=horizon)
            )
            merged = pd.merge_asof(
                merged.sort_values("Timestamp"),
                target.sort_values("lookup_timestamp"),
                left_on="Timestamp",
                right_on="lookup_timestamp",
                direction="nearest",
                tolerance=tolerance,
            ).drop(columns=["lookup_timestamp"])
        frames.append(merged)

    if not frames:
        return pd.DataFrame()

    supervised = pd.concat(frames, ignore_index=True)
    target_cols = target_columns(horizons)
    return supervised.dropna(subset=target_cols).reset_index(drop=True)


def target_columns(horizons: list[int]) -> list[str]:
    cols: list[str] = []
    for horizon in horizons:
        cols.extend([f"target_lat_{horizon}h", f"target_lon_{horizon}h"])
    return cols


def make_model(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=max(args.n_estimators, 1),
                    max_depth=args.max_depth,
                    min_samples_leaf=max(args.min_samples_leaf, 1),
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def group_train_test_split(
    data: pd.DataFrame,
    groups: pd.Series,
    test_size: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=min(max(test_size, 0.05), 0.5),
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(splitter.split(data, groups=groups))
    return data.iloc[train_idx].copy(), data.iloc[test_idx].copy()


def sample_rows_by_group(data: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(data) <= max_rows:
        return data

    rng = np.random.default_rng(RANDOM_STATE)
    group_sizes = data.groupby("MMSI").size()
    shuffled_groups = rng.permutation(group_sizes.index.to_numpy())
    selected: list[str] = []
    total = 0
    for group in shuffled_groups:
        selected.append(str(group))
        total += int(group_sizes.loc[group])
        if total >= max_rows:
            break
    return data.loc[data["MMSI"].isin(selected)].copy()


def fit_and_evaluate(
    supervised: pd.DataFrame,
    horizons: list[int],
    args: argparse.Namespace,
) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    train_df, test_df = group_train_test_split(supervised, supervised["MMSI"], args.test_size)
    fit_train_df = sample_rows_by_group(train_df, args.max_train_rows)

    x_train = fit_train_df[FEATURE_COLUMNS]
    y_train = fit_train_df[target_columns(horizons)]
    x_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[target_columns(horizons)]

    model = make_model(args)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)

    mean_errors = horizon_errors_km(y_test.to_numpy(), pred, horizons)
    mae_by_target = {
        col: float(mean_absolute_error(y_test[col], pred[:, idx]))
        for idx, col in enumerate(target_columns(horizons))
    }
    leakage_overlap = len(set(train_df["MMSI"]).intersection(set(test_df["MMSI"])))

    metrics = {
        "model_name": "random_forest_regressor",
        "display_name": "RandomForestRegressor",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "evaluation_method": "MMSI GroupShuffleSplit holdout",
        "horizons_hours": horizons,
        "rows": {
            "supervised": int(len(supervised)),
            "train": int(len(train_df)),
            "train_used_for_fit": int(len(fit_train_df)),
            "test": int(len(test_df)),
        },
        "groups": {
            "train": int(train_df["MMSI"].nunique()),
            "test": int(test_df["MMSI"].nunique()),
            "overlap": int(leakage_overlap),
        },
        "holdout_mean_error_km": mean_errors,
        "holdout_mae_degrees": mae_by_target,
        "feature_columns": FEATURE_COLUMNS,
    }
    return model, metrics, fit_train_df


def horizon_errors_km(y_true: np.ndarray, y_pred: np.ndarray, horizons: list[int]) -> dict[str, float]:
    errors: dict[str, float] = {}
    for idx, horizon in enumerate(horizons):
        lat_idx = idx * 2
        lon_idx = lat_idx + 1
        distances = haversine_km(
            y_true[:, lat_idx],
            y_true[:, lon_idx],
            y_pred[:, lat_idx],
            y_pred[:, lon_idx],
        )
        errors[f"{horizon}h"] = float(np.nanmean(distances))
    return errors


def haversine_km(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> np.ndarray:
    lat1_arr = np.radians(np.asarray(lat1, dtype=float))
    lon1_arr = np.radians(np.asarray(lon1, dtype=float))
    lat2_arr = np.radians(np.asarray(lat2, dtype=float))
    lon2_arr = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2_arr - lat1_arr
    dlon = lon2_arr - lon1_arr
    value = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_arr) * np.cos(lat2_arr) * np.sin(dlon / 2.0) ** 2
    )
    return 6371.0088 * 2.0 * np.arcsin(np.minimum(1.0, np.sqrt(value)))


def refit_for_deploy(
    supervised: pd.DataFrame,
    horizons: list[int],
    args: argparse.Namespace,
) -> Pipeline:
    fit_df = sample_rows_by_group(supervised, args.max_train_rows)
    model = make_model(args)
    model.fit(fit_df[FEATURE_COLUMNS], fit_df[target_columns(horizons)])
    return model


def latest_position_predictions(
    model: Pipeline,
    feature_data: pd.DataFrame,
    horizons: list[int],
    metrics: dict[str, Any],
) -> pd.DataFrame:
    latest = (
        feature_data.sort_values(["MMSI", "Timestamp"], kind="mergesort")
        .groupby("MMSI", as_index=False)
        .tail(1)
        .copy()
    )
    pred = model.predict(latest[FEATURE_COLUMNS])

    output = latest[["MMSI", "Timestamp", "Latitude", "Longitude"]].rename(
        columns={
            "Timestamp": "start_timestamp",
            "Latitude": "start_lat",
            "Longitude": "start_lon",
        }
    )
    for idx, horizon in enumerate(horizons):
        output[f"pred_lat_{horizon}h"] = np.clip(pred[:, idx * 2], -90, 90)
        output[f"pred_lon_{horizon}h"] = np.clip(pred[:, idx * 2 + 1], -180, 180)

    errors = metrics.get("holdout_mean_error_km", {})
    numeric_errors = [float(value) for value in errors.values() if value is not None]
    output["mean_error_km"] = float(np.mean(numeric_errors)) if numeric_errors else np.nan
    output["generated_at"] = datetime.now(timezone.utc).isoformat()
    return output.sort_values("MMSI", ignore_index=True)


def save_json(path: Path, data: dict[str, Any]) -> None:
    def default(value: Any) -> Any:
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=default), encoding="utf-8")


def main() -> None:
    args = parse_args()
    horizons = sorted({int(value) for value in args.horizons if int(value) > 0})
    if not horizons:
        raise ValueError("At least one positive horizon is required.")

    points = add_motion_features(load_ais_points(args.data.resolve()))
    supervised = make_supervised_rows(
        points,
        horizons=horizons,
        tolerance_minutes=args.tolerance_minutes,
    )
    if supervised.empty:
        raise RuntimeError("No supervised rows could be built for the requested horizons.")

    _, metrics, _ = fit_and_evaluate(supervised, horizons, args)
    deploy_model = refit_for_deploy(supervised, horizons, args)
    prediction_rows = latest_position_predictions(deploy_model, points, horizons, metrics)

    bundle = {
        "model_name": metrics["model_name"],
        "display_name": metrics["display_name"],
        "estimator": deploy_model,
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": target_columns(horizons),
        "horizons_hours": horizons,
        "metrics": metrics,
    }

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out.resolve(), compress=3)
    save_json(args.metrics_out.resolve(), metrics)
    prediction_rows.to_csv(args.predictions_out.resolve(), index=False, encoding="utf-8-sig")

    print(f"Saved future position model: {args.model_out.resolve()}")
    print(f"Saved future position metrics: {args.metrics_out.resolve()}")
    print(f"Saved future position predictions: {args.predictions_out.resolve()}")
    for horizon, error in metrics["holdout_mean_error_km"].items():
        print(f"- {horizon}: mean error {error:.3f} km")
    print(f"Group leakage check: overlap={metrics['groups']['overlap']} MMSI")


if __name__ == "__main__":
    main()
