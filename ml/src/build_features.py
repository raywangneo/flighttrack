"""Join processed BTS monthly parquet files with per-airport weather data
and engineer the feature table used for training.

Also emits weather_climatology.parquet: per-airport, per-month average of
each weather feature, used by the backend to approximate weather for
flights scheduled too far in the future for a real forecast.
"""

from __future__ import annotations

import pandas as pd

from common import CLIMATOLOGY_PATH, PROCESSED_DIR, WEATHER_HOURLY_VARS, WEATHER_RAW_DIR

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
WEATHER_WINDOW_HOURS = 2  # +/- window around scheduled local time to average weather over


def _parse_crs_time(series: pd.Series) -> pd.Series:
    # BTS CRSDepTime/CRSArrTime are integers like 830 or 1705 (HHMM, no colon).
    hhmm = series.astype(int).astype(str).str.zfill(4)
    return hhmm.str[:2].astype(int)


def load_flights() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("bts_*.parquet"))
    if not files:
        raise RuntimeError("No bts_*.parquet files found — run download_bts.py first.")
    frames = [pd.read_parquet(f) for f in files]
    df = pd.concat(frames, ignore_index=True)

    # Drop cancelled/diverted flights — ArrDel15 is null for these and they're
    # a categorically different outcome from "delayed".
    df = df[(df["Cancelled"] == 0) & (df["Diverted"] == 0)].copy()
    df = df.dropna(subset=["ArrDel15"])

    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    df["sched_dep_hour"] = _parse_crs_time(df["CRSDepTime"])
    df["sched_arr_hour"] = _parse_crs_time(df["CRSArrTime"])
    return df


def load_weather_lookup() -> dict[str, pd.DataFrame]:
    lookup = {}
    for path in WEATHER_RAW_DIR.glob("*.parquet"):
        iata = path.stem
        w = pd.read_parquet(path)
        w = w.set_index("time").sort_index()
        lookup[iata] = w
    if not lookup:
        raise RuntimeError("No weather parquet files found — run download_weather.py first.")
    return lookup


def _weather_at(
    weather_lookup: dict[str, pd.DataFrame], iata: str, ts: pd.Timestamp
) -> dict[str, float]:
    w = weather_lookup.get(iata)
    if w is None:
        return {f"{v}": pd.NA for v in WEATHER_HOURLY_VARS}
    window = w.loc[
        ts - pd.Timedelta(hours=WEATHER_WINDOW_HOURS) : ts + pd.Timedelta(hours=WEATHER_WINDOW_HOURS)
    ]
    if window.empty:
        return {v: pd.NA for v in WEATHER_HOURLY_VARS}
    return {v: window[v].mean() for v in WEATHER_HOURLY_VARS if v in window.columns}


def attach_weather(df: pd.DataFrame, weather_lookup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    dep_ts = df["FlightDate"] + pd.to_timedelta(df["sched_dep_hour"], unit="h")
    arr_ts = df["FlightDate"] + pd.to_timedelta(df["sched_arr_hour"], unit="h")

    origin_weather = [
        _weather_at(weather_lookup, o, t) for o, t in zip(df["Origin"], dep_ts)
    ]
    dest_weather = [
        _weather_at(weather_lookup, d, t) for d, t in zip(df["Dest"], arr_ts)
    ]

    origin_df = pd.DataFrame(origin_weather, index=df.index).add_prefix("origin_")
    dest_df = pd.DataFrame(dest_weather, index=df.index).add_prefix("dest_")

    return pd.concat([df, origin_df, dest_df], axis=1)


def build_climatology(weather_lookup: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for iata, w in weather_lookup.items():
        monthly = w.groupby(w.index.month)[WEATHER_HOURLY_VARS].mean()
        for month, row in monthly.iterrows():
            rows.append({"iata": iata, "month": month, **row.to_dict()})
    clim = pd.DataFrame(rows)
    CLIMATOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(CLIMATOLOGY_PATH, index=False)
    print(f"wrote {CLIMATOLOGY_PATH} ({len(clim)} airport-month rows)")
    return clim


def build_features() -> pd.DataFrame:
    print("loading flights...")
    flights = load_flights()
    print(f"  {len(flights):,} flights after filtering cancelled/diverted")

    print("loading weather...")
    weather_lookup = load_weather_lookup()

    print("building climatology...")
    build_climatology(weather_lookup)

    print("joining weather to flights (this can take a few minutes)...")
    features = attach_weather(flights, weather_lookup)

    for col in ["Reporting_Airline", "Origin", "Dest"]:
        features[col] = features[col].astype("category")

    keep_cols = [
        "FlightDate",
        "Reporting_Airline",
        "Origin",
        "Dest",
        "sched_dep_hour",
        "sched_arr_hour",
        "DayOfWeek",
        "Month",
        "Distance",
        "CRSElapsedTime",
        "ArrDel15",
    ] + [f"origin_{v}" for v in WEATHER_HOURLY_VARS] + [f"dest_{v}" for v in WEATHER_HOURLY_VARS]
    features = features[keep_cols]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    print(f"wrote {FEATURES_PATH} ({len(features):,} rows, {len(features.columns)} columns)")
    return features


if __name__ == "__main__":
    build_features()
