from __future__ import annotations

import json
from datetime import datetime
from functools import lru_cache
from pathlib import Path

import pandas as pd
import xgboost as xgb

from .schemas import PredictRequest, PredictResponse
from .weather import get_weather

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_VERSION = "v1"


@lru_cache
def _load():
    booster = xgb.Booster()
    booster.load_model(str(MODEL_DIR / "model.json"))
    with open(MODEL_DIR / "feature_metadata.json") as f:
        metadata = json.load(f)
    return booster, metadata


def warm_up():
    """Call once at startup so the first real request isn't the one paying
    the model/metadata load cost."""
    _load()


def get_supported_airlines() -> list[str]:
    _, metadata = _load()
    return metadata["category_maps"]["Reporting_Airline"]


def _build_feature_row(req: PredictRequest, metadata: dict) -> pd.DataFrame:
    dep_dt = datetime.fromisoformat(req.scheduled_departure)

    origin_weather, weather_source = get_weather(req.origin, dep_dt)
    dest_weather, _ = get_weather(req.dest, dep_dt)

    row = {
        "Reporting_Airline": req.airline.upper(),
        "Origin": req.origin.upper(),
        "Dest": req.dest.upper(),
        "sched_dep_hour": dep_dt.hour,
        "sched_arr_hour": dep_dt.hour,  # arrival hour unknown at request time; departure hour is the best proxy available
        "DayOfWeek": dep_dt.isoweekday(),
        "Month": dep_dt.month,
        "Distance": metadata["numeric_medians"]["Distance"],
        "CRSElapsedTime": metadata["numeric_medians"]["CRSElapsedTime"],
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
    prob = float(booster.predict(dmatrix)[0])

    threshold = metadata.get("decision_threshold", 0.5)
    if prob < 0.25:
        risk_level = "low"
    elif prob < 0.5:
        risk_level = "medium"
    else:
        risk_level = "high"

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
        delay_probability=round(prob, 4),
        delayed_prediction=prob >= threshold,
        risk_level=risk_level,
        weather_source=weather_source,
        model_version=MODEL_VERSION,
        caveats=caveats,
    )
