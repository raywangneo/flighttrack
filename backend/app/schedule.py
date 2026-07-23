"""Flight-schedule lookup — search by flight number or by route+time-of-day.

Built from ml/src/build_schedule_reference.py's output: distinct
(airline, flight_number, origin, dest, rounded departure time) combos seen
in the most recent ~3 months of BTS data, each tagged with a days_mask
bitmask of which weekdays that slot has historically operated on.

days_mask convention: bit0=Monday .. bit6=Sunday (i.e. bit (isoweekday-1)),
matching predict.py's dep_dt.isoweekday() and the frontend's
src/lib/schedule.ts remap of JS's 0=Sunday-based Date.getDay(). Keep these
three in sync if this convention ever changes.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd

SCHEDULE_REFERENCE_PATH = Path(__file__).resolve().parent / "model" / "schedule_reference.parquet"

FLIGHT_NUMBER_RE = re.compile(r"^([A-Za-z]{2})\s?0*(\d{1,4})$")
BAND_WIDTHS_HOURS = [1, 2, 3]


@lru_cache
def load_schedule_reference() -> pd.DataFrame:
    return pd.read_parquet(SCHEDULE_REFERENCE_PATH)


def _row_to_itinerary(row: pd.Series) -> dict:
    return {
        "airline": row["airline"],
        "flight_number": row["flight_number"],
        "origin": row["origin"],
        "dest": row["dest"],
        "dep_time_minutes": int(row["dep_time_minutes"]),
        "arr_time_minutes": int(row["arr_time_minutes"]),
        "distance_miles": float(row["distance_miles"]),
        "elapsed_minutes": float(row["elapsed_minutes"]),
        "days_mask": int(row["days_mask"]),
        "sample_count": int(row["sample_count"]),
    }


def parse_flight_number(query: str) -> tuple[str, str] | None:
    match = FLIGHT_NUMBER_RE.match(query.strip())
    if not match:
        return None
    airline, number = match.groups()
    return airline.upper(), number


def search_by_flight_number(query: str) -> tuple[str, str, list[dict]] | None:
    parsed = parse_flight_number(query)
    if parsed is None:
        return None
    airline, number = parsed

    ref = load_schedule_reference()
    matches = ref[(ref["airline"] == airline) & (ref["flight_number"] == number)]
    itineraries = [_row_to_itinerary(r) for _, r in matches.iterrows()]
    return airline, number, itineraries


def search_by_route(origin: str, dest: str, time_of_day_minutes: int) -> tuple[list[dict], int]:
    ref = load_schedule_reference()
    candidates = ref[(ref["origin"] == origin.upper()) & (ref["dest"] == dest.upper())]

    for band_hours in BAND_WIDTHS_HOURS:
        window = band_hours * 60
        matches = candidates[
            (candidates["dep_time_minutes"] >= time_of_day_minutes - window)
            & (candidates["dep_time_minutes"] <= time_of_day_minutes + window)
        ]
        if not matches.empty:
            matches = matches.sort_values("dep_time_minutes")
            return [_row_to_itinerary(r) for _, r in matches.iterrows()], band_hours

    return [], 3
