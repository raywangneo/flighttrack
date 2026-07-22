"""Join processed BTS monthly parquet files with per-airport weather data
and engineer the feature table used for training.

Uses pandas merge_asof (vectorized nearest-timestamp join) rather than a
per-row Python loop — with ~15M flight rows, a per-row .loc lookup would
take hours; merge_asof does the equivalent join in seconds.

Also emits weather_climatology.parquet: per-airport, per-month average of
each weather feature, used by the backend to approximate weather for
flights scheduled too far in the future for a real forecast.
"""

from __future__ import annotations

import pandas as pd

from common import CLIMATOLOGY_PATH, PROCESSED_DIR, WEATHER_HOURLY_VARS, WEATHER_RAW_DIR

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
WEATHER_WINDOW_HOURS = 2  # max distance to the nearest hourly weather reading


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
    return df.reset_index(drop=True)


def load_weather_all() -> pd.DataFrame:
    frames = []
    for path in sorted(WEATHER_RAW_DIR.glob("*.parquet")):
        w = pd.read_parquet(path)
        w["iata"] = path.stem
        frames.append(w)
    if not frames:
        raise RuntimeError("No weather parquet files found — run download_weather.py first.")
    weather_all = pd.concat(frames, ignore_index=True)
    # merge_asof requires the "on" column sorted globally (the "by" grouping
    # only restricts matches, it doesn't need its own sort order).
    weather_all = weather_all.sort_values("time").reset_index(drop=True)
    return weather_all


def _asof_join(iata: pd.Series, ts: pd.Series, weather_all: pd.DataFrame, prefix: str) -> pd.DataFrame:
    keys = pd.DataFrame({"iata": iata.to_numpy(), "time": ts.to_numpy()})
    keys["_row"] = range(len(keys))
    keys_sorted = keys.sort_values("time")

    merged = pd.merge_asof(
        keys_sorted,
        weather_all,
        on="time",
        by="iata",
        direction="nearest",
        tolerance=pd.Timedelta(hours=WEATHER_WINDOW_HOURS),
    )
    merged = merged.sort_values("_row").reset_index(drop=True)
    return merged[WEATHER_HOURLY_VARS].add_prefix(prefix)


def attach_weather(df: pd.DataFrame, weather_all: pd.DataFrame) -> pd.DataFrame:
    dep_ts = df["FlightDate"] + pd.to_timedelta(df["sched_dep_hour"], unit="h")
    arr_ts = df["FlightDate"] + pd.to_timedelta(df["sched_arr_hour"], unit="h")

    print("  joining origin weather...")
    origin_df = _asof_join(df["Origin"], dep_ts, weather_all, "origin_")
    print("  joining destination weather...")
    dest_df = _asof_join(df["Dest"], arr_ts, weather_all, "dest_")

    return pd.concat([df, origin_df, dest_df], axis=1)


def build_climatology(weather_all: pd.DataFrame) -> pd.DataFrame:
    tmp = weather_all.copy()
    tmp["month"] = tmp["time"].dt.month
    clim = tmp.groupby(["iata", "month"])[WEATHER_HOURLY_VARS].mean().reset_index()
    CLIMATOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(CLIMATOLOGY_PATH, index=False)
    print(f"wrote {CLIMATOLOGY_PATH} ({len(clim)} airport-month rows)")
    return clim


def build_features() -> pd.DataFrame:
    print("loading flights...")
    flights = load_flights()
    print(f"  {len(flights):,} flights after filtering cancelled/diverted")

    print("loading weather...")
    weather_all = load_weather_all()
    print(f"  {len(weather_all):,} hourly weather rows across {weather_all['iata'].nunique()} airports")

    print("building climatology...")
    build_climatology(weather_all)

    print("joining weather to flights...")
    features = attach_weather(flights, weather_all)

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
