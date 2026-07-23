from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .schemas import BucketProbabilities, PredictRequest, PredictResponse
from .weather import get_weather

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_VERSION = "v2-bucketed"


@lru_cache
def _load():
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "model.json"))
    with open(MODEL_DIR / "feature_metadata.json") as f:
        metadata = json.load(f)
    return booster, metadata


@lru_cache
def _load_traffic_climatology() -> pd.DataFrame:
    return pd.read_parquet(MODEL_DIR / "traffic_climatology.parquet")


@lru_cache
def _load_performance_climatology() -> pd.DataFrame:
    return pd.read_parquet(MODEL_DIR / "performance_climatology.parquet")


def warm_up():
    """Call once at startup so the first real request isn't the one paying
    the model/metadata/climatology load cost."""
    _load()
    _load_traffic_climatology()
    _load_performance_climatology()


def get_supported_airlines() -> list[str]:
    _, metadata = _load()
    return metadata["category_maps"]["Reporting_Airline"]


def _lookup_traffic(iata: str, day_of_week: int, hour: int, column: str) -> float | None:
    clim = _load_traffic_climatology()
    row = clim[(clim["iata"] == iata) & (clim["DayOfWeek"] == day_of_week) & (clim["hour"] == hour)]
    return float(row.iloc[0][column]) if not row.empty else None


def _lookup_ontime_rate(airline: str, origin: str) -> float | None:
    clim = _load_performance_climatology()
    row = clim[(clim["airline"] == airline) & (clim["iata"] == origin)]
    return float(row.iloc[0]["ontime_rate"]) if not row.empty else None


def _apply_calibration(raw_probs: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Mirrors ml/src/train.py's apply_calibration for the multi-class
    one-vs-rest isotonic calibration — reimplemented here (rather than
    imported) since the backend ships independently of the ml/ package."""
    if not calibration:
        return raw_probs
    calibrated = np.zeros_like(raw_probs, dtype=float)
    for k, cal in enumerate(calibration["per_class"]):
        calibrated[k] = np.interp(raw_probs[k], cal["x_thresholds"], cal["y_thresholds"])
    total = max(float(calibrated.sum()), 1e-9)
    return calibrated / total


def _build_feature_row(req: PredictRequest, metadata: dict) -> tuple[pd.DataFrame, str]:
    dep_dt = datetime.fromisoformat(req.scheduled_departure)
    origin = req.origin.upper()
    dest = req.dest.upper()
    airline = req.airline.upper()

    origin_weather, weather_source = get_weather(origin, dep_dt)
    dest_weather, _ = get_weather(dest, dep_dt)

    row = {
        "Reporting_Airline": airline,
        "Origin": origin,
        "Dest": dest,
        "sched_dep_hour": dep_dt.hour,
        "sched_arr_hour": dep_dt.hour,  # arrival hour unknown at request time; departure hour is the best proxy available
        "DayOfWeek": dep_dt.isoweekday(),
        "Month": dep_dt.month,
        "Distance": metadata["numeric_medians"]["Distance"],
        "CRSElapsedTime": metadata["numeric_medians"]["CRSElapsedTime"],
        "origin_hourly_traffic": _lookup_traffic(origin, dep_dt.isoweekday(), dep_dt.hour, "avg_dep_traffic"),
        "dest_hourly_traffic": _lookup_traffic(dest, dep_dt.isoweekday(), dep_dt.hour, "avg_arr_traffic"),
        "rolling_ontime_rate": _lookup_ontime_rate(airline, origin),
    }
    for v in ["precipitation", "rain", "snowfall", "wind_speed_10m", "wind_gusts_10m", "cloud_cover", "cloud_cover_low"]:
        row[f"origin_{v}"] = origin_weather.get(v) or metadata["numeric_medians"].get(f"origin_{v}")
        row[f"dest_{v}"] = dest_weather.get(v) or metadata["numeric_medians"].get(f"dest_{v}")

    df = pd.DataFrame([row])

    for col, categories in metadata["category_maps"].items():
        df[col] = pd.Categorical(df[col], categories=categories)

    for col in metadata["numeric_cols"]:
        if col not in df.columns:
            df[col] = metadata["numeric_medians"].get(col)
        df[col] = df[col].fillna(metadata["numeric_medians"].get(col))

    return df[metadata["feature_order"]], weather_source


def predict(req: PredictRequest) -> PredictResponse:
    booster, metadata = _load()
    features, weather_source = _build_feature_row(req, metadata)

    dmatrix = xgb.DMatrix(features, enable_categorical=True)
    raw_probs = booster.predict(dmatrix)[0]  # shape (5,) for multi:softprob, single row
    probs = _apply_calibration(raw_probs, metadata.get("calibration"))

    bucket_labels = metadata["bucket_labels"]
    predicted_idx = int(np.argmax(probs))
    predicted_bucket = bucket_labels[predicted_idx]
    bucket_probability = float(probs[predicted_idx])
    bucket_probabilities = BucketProbabilities(**{label: round(float(p), 4) for label, p in zip(bucket_labels, probs)})

    caveats = []
    if weather_source == "historical_average":
        caveats.append(
            "Forecast unavailable this far out; using seasonal average weather for this route/date."
        )
    if req.origin.upper() not in metadata["category_maps"]["Origin"]:
        caveats.append(f"Origin airport '{req.origin}' was not in the training data; prediction may be less reliable.")
    if req.dest.upper() not in metadata["category_maps"]["Dest"]:
        caveats.append(f"Destination airport '{req.dest}' was not in the training data; prediction may be less reliable.")
    if req.airline.upper() not in metadata["category_maps"]["Reporting_Airline"]:
        caveats.append(f"Airline '{req.airline}' was not in the training data; prediction may be less reliable.")

    return PredictResponse(
        predicted_bucket=predicted_bucket,
        bucket_probability=round(bucket_probability, 4),
        bucket_probabilities=bucket_probabilities,
        weather_source=weather_source,
        model_version=MODEL_VERSION,
        caveats=caveats,
    )
