"""Weather lookups for the /predict endpoint.

Addresses the training/inference weather gap: historical weather (used at
training time) doesn't exist for a flight scheduled next week. If the
scheduled departure is within Open-Meteo's ~16-day forecast horizon, fetch
a real forecast. Otherwise (the common case — most flights are booked
weeks/months out), fall back to precomputed seasonal climatology by
airport+month, and flag the response with a caveat so the app can be
honest about reduced accuracy rather than presenting false precision.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from .airports import get_lat_lon

CLIMATOLOGY_PATH = Path(__file__).resolve().parent / "model" / "weather_climatology.parquet"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"
FORECAST_HORIZON_DAYS = 16
WEATHER_VARS = [
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
]

_forecast_cache: dict[tuple[str, int], tuple[float, dict]] = {}
_CACHE_TTL_SECONDS = 3600


@lru_cache
def load_climatology() -> pd.DataFrame:
    return pd.read_parquet(CLIMATOLOGY_PATH)


def _fetch_forecast(iata: str, target_dt: datetime) -> dict:
    cache_key = (iata, target_dt.hour, target_dt.date().isoformat())
    now = time.time()
    cached = _forecast_cache.get(cache_key)
    if cached and now - cached[0] < _CACHE_TTL_SECONDS:
        return cached[1]

    lat, lon = get_lat_lon(iata)
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "auto",
        "start_date": target_dt.date().isoformat(),
        "end_date": target_dt.date().isoformat(),
    }
    resp = requests.get(FORECAST_API, params=params, timeout=15)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]

    times = hourly["time"]
    target_iso_hour = target_dt.strftime("%Y-%m-%dT%H:00")
    if target_iso_hour in times:
        idx = times.index(target_iso_hour)
    else:
        idx = min(range(len(times)), key=lambda i: abs(i - target_dt.hour))

    result = {v: hourly[v][idx] for v in WEATHER_VARS if v in hourly}
    _forecast_cache[cache_key] = (now, result)
    return result


def _climatology_lookup(iata: str, month: int) -> dict:
    clim = load_climatology()
    row = clim[(clim["iata"] == iata) & (clim["month"] == month)]
    if row.empty:
        return {v: None for v in WEATHER_VARS}
    row = row.iloc[0]
    return {v: float(row[v]) for v in WEATHER_VARS if v in row}


def get_weather(iata: str, target_dt: datetime) -> tuple[dict, Literal["forecast", "historical_average"]]:
    days_out = (target_dt.date() - datetime.now().date()).days
    if 0 <= days_out <= FORECAST_HORIZON_DAYS:
        try:
            return _fetch_forecast(iata, target_dt), "forecast"
        except requests.RequestException:
            pass  # fall through to climatology if the forecast API is unavailable
    return _climatology_lookup(iata, target_dt.month), "historical_average"
