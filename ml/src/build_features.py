"""Join processed BTS monthly parquet files with per-airport weather data
and engineer the feature table used for training.

Uses pandas merge_asof (vectorized nearest-timestamp join) rather than a
per-row Python loop — with ~15M flight rows, a per-row .loc lookup would
take hours; merge_asof does the equivalent join in seconds.

v2 adds three "Tier 1" features on top of schedule + weather:
  - upstream delay: was the specific aircraft (by Tail_Number) already late
    arriving from its previous leg that day, and how much scheduled buffer
    does it have before this departure. NOTE: this is a training-only
    signal in the deployed sense — a future flight's aircraft assignment is
    unknowable in advance, so the backend always sends first_flight_of_day=1
    (i.e. "unknown") at serving time. It's included so the model can learn
    from it when available and to measure its offline contribution honestly.
  - airport congestion: how many other flights are scheduled at the same
    airport in the same hour. Knowable in advance via typical traffic
    patterns, so this DOES carry through to serving (via traffic_climatology).
  - rolling recent performance: trailing 30-day on-time rate for this
    airline/origin combo, computed leakage-safe (only using days strictly
    before the flight's own date). Serving uses the airline/origin's overall
    historical rate as a proxy (via performance_climatology), since we don't
    run a live recent-performance data pipeline.

Also emits weather_climatology.parquet, traffic_climatology.parquet, and
performance_climatology.parquet — small lookup tables the backend uses to
approximate these features for future flights.
"""

from __future__ import annotations

import pandas as pd

from common import (
    CLIMATOLOGY_PATH,
    PERFORMANCE_CLIMATOLOGY_PATH,
    PROCESSED_DIR,
    TRAFFIC_CLIMATOLOGY_PATH,
    WEATHER_HOURLY_VARS,
    WEATHER_RAW_DIR,
)

FEATURES_PATH = PROCESSED_DIR / "features.parquet"
WEATHER_WINDOW_HOURS = 2  # max distance to the nearest hourly weather reading
ROLLING_WINDOW = "30D"


def _parse_crs_time(series: pd.Series) -> pd.Series:
    # BTS CRSDepTime/CRSArrTime are integers like 830 or 1705 (HHMM, no colon).
    hhmm = series.astype(int).astype(str).str.zfill(4)
    return hhmm.str[:2].astype(int)


def _crs_time_to_minutes(series: pd.Series) -> pd.Series:
    hhmm = series.astype(int).astype(str).str.zfill(4)
    return hhmm.str[:2].astype(int) * 60 + hhmm.str[2:].astype(int)


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


def attach_upstream_delay(df: pd.DataFrame) -> pd.DataFrame:
    """Was this specific aircraft's previous leg today already late, and how
    much scheduled ground time does it have before this departure. Chains
    are built per (Tail_Number, FlightDate) — doesn't cross midnight into
    the previous calendar day, a reasonable simplification for v1."""
    df = df.copy()
    df["_dep_min"] = _crs_time_to_minutes(df["CRSDepTime"])
    df["_arr_min"] = _crs_time_to_minutes(df["CRSArrTime"])

    has_tail = df["Tail_Number"].notna() & (df["Tail_Number"] != "")
    chainable = df[has_tail].sort_values(["Tail_Number", "FlightDate", "_dep_min"])

    grouped = chainable.groupby(["Tail_Number", "FlightDate"], observed=True)
    prior_arr_delay = grouped["ArrDelayMinutes"].shift(1)
    prior_arr_min = grouped["_arr_min"].shift(1)
    turnaround = chainable["_dep_min"] - prior_arr_min
    turnaround = turnaround.where(turnaround >= 0)  # negative => date-boundary quirk, treat as unknown

    df["prior_arr_delay"] = pd.Series(prior_arr_delay, index=chainable.index).reindex(df.index)
    df["scheduled_turnaround_minutes"] = pd.Series(turnaround, index=chainable.index).reindex(df.index)

    df["first_flight_of_day"] = df["prior_arr_delay"].isna().astype(int)
    df["prior_arr_delay"] = df["prior_arr_delay"].fillna(0.0)
    median_turnaround = df["scheduled_turnaround_minutes"].median()
    df["scheduled_turnaround_minutes"] = df["scheduled_turnaround_minutes"].fillna(median_turnaround)

    return df.drop(columns=["_dep_min", "_arr_min"])


def attach_congestion(df: pd.DataFrame) -> pd.DataFrame:
    """How many other flights are scheduled at the same airport in the same
    hour — a proxy for structural congestion independent of weather."""
    df = df.copy()
    origin_counts = (
        df.groupby(["Origin", "FlightDate", "sched_dep_hour"], observed=True)
        .size()
        .rename("origin_hourly_traffic")
    )
    df = df.merge(origin_counts, on=["Origin", "FlightDate", "sched_dep_hour"], how="left")

    dest_counts = (
        df.groupby(["Dest", "FlightDate", "sched_arr_hour"], observed=True)
        .size()
        .rename("dest_hourly_traffic")
    )
    df = df.merge(dest_counts, on=["Dest", "FlightDate", "sched_arr_hour"], how="left")
    return df


def build_traffic_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Average scheduled traffic by (airport, day-of-week, hour) — used by
    the backend to approximate congestion for future flights, since we
    don't have access to real future published schedules."""
    dep_daily = (
        df.groupby(["Origin", "FlightDate", "DayOfWeek", "sched_dep_hour"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    dep_avg = (
        dep_daily.groupby(["Origin", "DayOfWeek", "sched_dep_hour"], observed=True)["n"]
        .mean()
        .reset_index()
        .rename(columns={"Origin": "iata", "sched_dep_hour": "hour", "n": "avg_dep_traffic"})
    )

    arr_daily = (
        df.groupby(["Dest", "FlightDate", "DayOfWeek", "sched_arr_hour"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    arr_avg = (
        arr_daily.groupby(["Dest", "DayOfWeek", "sched_arr_hour"], observed=True)["n"]
        .mean()
        .reset_index()
        .rename(columns={"Dest": "iata", "sched_arr_hour": "hour", "n": "avg_arr_traffic"})
    )

    clim = pd.merge(dep_avg, arr_avg, on=["iata", "DayOfWeek", "hour"], how="outer").fillna(0)
    TRAFFIC_CLIMATOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(TRAFFIC_CLIMATOLOGY_PATH, index=False)
    print(f"wrote {TRAFFIC_CLIMATOLOGY_PATH} ({len(clim)} rows)")
    return clim


def attach_rolling_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Trailing 30-day on-time rate for this (airline, origin) combo,
    computed leakage-safe: closed='left' excludes the flight's own day, so
    only strictly-prior data informs the feature."""
    df = df.copy()
    daily = (
        df.groupby(["Reporting_Airline", "Origin", "FlightDate"], observed=True)["ArrDel15"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["Reporting_Airline", "Origin", "FlightDate"])
        .set_index("FlightDate")
    )

    rolled = (
        daily.groupby(["Reporting_Airline", "Origin"], observed=True)[["sum", "count"]]
        .rolling(ROLLING_WINDOW, closed="left")
        .sum()
        .reset_index()
    )
    rolled["rolling_ontime_rate"] = 1 - (rolled["sum"] / rolled["count"].replace(0, pd.NA))

    df = df.merge(
        rolled[["Reporting_Airline", "Origin", "FlightDate", "rolling_ontime_rate"]],
        on=["Reporting_Airline", "Origin", "FlightDate"],
        how="left",
    )
    # No prior data yet (start of the dataset / brand-new route) -> fall back
    # to that airline/origin's overall rate, then the global rate as a last resort.
    overall_by_group = df.groupby(["Reporting_Airline", "Origin"], observed=True)["ArrDel15"].transform(
        lambda s: 1 - s.mean()
    )
    global_rate = 1 - df["ArrDel15"].mean()
    df["rolling_ontime_rate"] = df["rolling_ontime_rate"].fillna(overall_by_group).fillna(global_rate)
    return df


def build_performance_climatology(df: pd.DataFrame) -> pd.DataFrame:
    """Overall historical on-time rate by (airline, origin) — used by the
    backend as a proxy for 'recent performance' since serving a real live
    rolling window would need an operational data pipeline we don't run."""
    clim = (
        df.groupby(["Reporting_Airline", "Origin"], observed=True)["ArrDel15"]
        .mean()
        .reset_index()
    )
    clim["ontime_rate"] = 1 - clim["ArrDel15"]
    clim = clim.drop(columns=["ArrDel15"]).rename(columns={"Reporting_Airline": "airline", "Origin": "iata"})
    PERFORMANCE_CLIMATOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    clim.to_parquet(PERFORMANCE_CLIMATOLOGY_PATH, index=False)
    print(f"wrote {PERFORMANCE_CLIMATOLOGY_PATH} ({len(clim)} airline-airport rows)")
    return clim


def build_features() -> pd.DataFrame:
    print("loading flights...")
    flights = load_flights()
    print(f"  {len(flights):,} flights after filtering cancelled/diverted")

    print("loading weather...")
    weather_all = load_weather_all()
    print(f"  {len(weather_all):,} hourly weather rows across {weather_all['iata'].nunique()} airports")

    print("building weather climatology...")
    build_climatology(weather_all)

    print("joining weather to flights...")
    features = attach_weather(flights, weather_all)

    print("computing upstream aircraft delay chains...")
    features = attach_upstream_delay(features)

    print("computing airport congestion...")
    features = attach_congestion(features)
    build_traffic_climatology(features)

    print("computing rolling airline/origin performance...")
    features = attach_rolling_performance(features)
    build_performance_climatology(features)

    for col in ["Reporting_Airline", "Origin", "Dest"]:
        features[col] = features[col].astype("category")

    keep_cols = (
        [
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
            "ArrDelayMinutes",
            "prior_arr_delay",
            "scheduled_turnaround_minutes",
            "first_flight_of_day",
            "origin_hourly_traffic",
            "dest_hourly_traffic",
            "rolling_ontime_rate",
        ]
        + [f"origin_{v}" for v in WEATHER_HOURLY_VARS]
        + [f"dest_{v}" for v in WEATHER_HOURLY_VARS]
    )
    features = features[keep_cols]

    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURES_PATH, index=False)
    print(f"wrote {FEATURES_PATH} ({len(features):,} rows, {len(features.columns)} columns)")
    return features


if __name__ == "__main__":
    build_features()
