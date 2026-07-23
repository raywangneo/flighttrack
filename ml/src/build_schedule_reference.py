"""Build a flight-schedule reference table from the most recent months of
BTS data — distinct from the ML training feature table, and used only for
letting the app's frontend search/browse real flight numbers and routes.

Unlike build_features.py, this does NOT filter out cancelled/diverted
flights: a cancelled flight is still evidence that slot exists on that
weekday, and filtering it out would create false negatives in day-of-week
compatibility for weather-heavy routes.

CRSDepTime/CRSArrTime are rounded to the nearest 15 minutes to absorb
day-to-day filing jitter for what is otherwise the same flight — flight
number is the primary disambiguator between distinct flights, not the
rounding, so 15 minutes is generous without risking silently merging a
genuine mid-quarter schedule change into one row.
"""

from __future__ import annotations

import pandas as pd
import pyarrow.parquet as pq

from common import (
    DEP_TIME_ROUNDING_MINUTES,
    FLIGHT_NUMBER_COL,
    PROCESSED_DIR,
    SCHEDULE_REFERENCE_MONTHS,
    SCHEDULE_REFERENCE_PATH,
)


def _crs_time_to_minutes(series: pd.Series) -> pd.Series:
    hhmm = series.astype(int).astype(str).str.zfill(4)
    return hhmm.str[:2].astype(int) * 60 + hhmm.str[2:].astype(int)


def _round_minutes(minutes: pd.Series, step: int) -> pd.Series:
    return ((minutes + step // 2) // step * step) % (24 * 60)


def load_recent_flights() -> pd.DataFrame:
    files = sorted(PROCESSED_DIR.glob("bts_*.parquet"))
    if not files:
        raise RuntimeError("No bts_*.parquet files found — run download_bts.py first.")

    candidates = []
    for path in files:
        schema_names = pq.ParquetFile(path).schema.names  # metadata-only, doesn't read row data
        if FLIGHT_NUMBER_COL in schema_names:
            candidates.append(path)

    if len(candidates) < SCHEDULE_REFERENCE_MONTHS:
        raise RuntimeError(
            f"Only {len(candidates)} month(s) have '{FLIGHT_NUMBER_COL}' "
            f"(need {SCHEDULE_REFERENCE_MONTHS}). Missing months: "
            f"{sorted(set(f.stem for f in files) - set(f.stem for f in candidates))}. "
            "Re-run download_bts.py --force for the recent months first."
        )

    recent = candidates[-SCHEDULE_REFERENCE_MONTHS:]
    print(f"using {[p.stem for p in recent]} for schedule reference")
    frames = [pd.read_parquet(p) for p in recent]
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=[FLIGHT_NUMBER_COL, "CRSDepTime", "CRSArrTime"])
    df["FlightDate"] = pd.to_datetime(df["FlightDate"])
    return df


def build_schedule_reference() -> pd.DataFrame:
    df = load_recent_flights()
    print(f"  {len(df):,} flights loaded")

    df["flight_number"] = df[FLIGHT_NUMBER_COL].astype("Int64").astype(str)
    df["dep_minutes"] = _crs_time_to_minutes(df["CRSDepTime"])
    df["arr_minutes"] = _crs_time_to_minutes(df["CRSArrTime"])
    df["dep_time_rounded"] = _round_minutes(df["dep_minutes"], DEP_TIME_ROUNDING_MINUTES)
    df["arr_time_rounded"] = _round_minutes(df["arr_minutes"], DEP_TIME_ROUNDING_MINUTES)

    group_cols = ["Reporting_Airline", "flight_number", "Origin", "Dest", "dep_time_rounded"]

    def _days_mask(day_of_week_values: pd.Series) -> int:
        mask = 0
        for d in day_of_week_values.unique():
            mask |= 1 << (int(d) - 1)  # BTS DayOfWeek: 1=Mon..7=Sun
        return mask

    grouped = df.groupby(group_cols, observed=True)
    schedule = grouped.agg(
        arr_time_rounded=("arr_time_rounded", "mean"),
        distance_miles=("Distance", "mean"),
        elapsed_minutes=("CRSElapsedTime", "mean"),
        sample_count=("FlightDate", "count"),
        first_seen=("FlightDate", "min"),
        last_seen=("FlightDate", "max"),
    ).reset_index()

    days_mask = grouped["DayOfWeek"].apply(_days_mask).rename("days_mask")
    schedule = schedule.merge(days_mask, on=group_cols)

    schedule["arr_time_rounded"] = schedule["arr_time_rounded"].round().astype(int)
    schedule = schedule.rename(
        columns={
            "Reporting_Airline": "airline",
            "Origin": "origin",
            "Dest": "dest",
            "dep_time_rounded": "dep_time_minutes",
            "arr_time_rounded": "arr_time_minutes",
        }
    )

    SCHEDULE_REFERENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_parquet(SCHEDULE_REFERENCE_PATH, index=False)
    print(f"wrote {SCHEDULE_REFERENCE_PATH} ({len(schedule):,} rows)")
    return schedule


if __name__ == "__main__":
    build_schedule_reference()
