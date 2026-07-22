"""Build ml/data/airports.csv (trimmed OurAirports lookup for IATA codes that
actually appear in the downloaded BTS months) and fetch historical hourly
weather for each of those airports from Open-Meteo's free Historical
Weather API, batching multiple airports per request and caching each
response so re-runs are idempotent.
"""

from __future__ import annotations

import argparse
import io
import time

import pandas as pd
import requests

from common import (
    AIRPORTS_CSV,
    HTTP_HEADERS,
    PROCESSED_DIR,
    WEATHER_HOURLY_VARS,
    WEATHER_RAW_DIR,
)

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
BATCH_SIZE = 5
BATCH_SLEEP_SECONDS = 20
MAX_RETRIES = 8


def collect_iata_codes() -> list[str]:
    codes: set[str] = set()
    for path in sorted(PROCESSED_DIR.glob("bts_*.parquet")):
        df = pd.read_parquet(path, columns=["Origin", "Dest"])
        codes.update(df["Origin"].unique())
        codes.update(df["Dest"].unique())
    if not codes:
        raise RuntimeError(
            "No bts_*.parquet files found in ml/data/processed/ — run download_bts.py first."
        )
    return sorted(codes)


def build_airports_csv(iata_codes: list[str]) -> pd.DataFrame:
    print(f"fetching OurAirports reference data for {len(iata_codes)} airports...")
    resp = requests.get(OURAIRPORTS_URL, headers=HTTP_HEADERS, timeout=60)
    resp.raise_for_status()
    all_airports = pd.read_csv(io.StringIO(resp.text), low_memory=False)

    trimmed = all_airports[all_airports["iata_code"].isin(iata_codes)][
        ["iata_code", "latitude_deg", "longitude_deg", "iso_region", "municipality"]
    ].rename(
        columns={
            "iata_code": "iata",
            "latitude_deg": "lat",
            "longitude_deg": "lon",
        }
    )
    trimmed = trimmed.drop_duplicates(subset="iata")

    missing = set(iata_codes) - set(trimmed["iata"])
    if missing:
        print(f"  warning: {len(missing)} IATA codes not found in OurAirports: {sorted(missing)}")

    AIRPORTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    trimmed.to_csv(AIRPORTS_CSV, index=False)
    print(f"  wrote {AIRPORTS_CSV} ({len(trimmed)} airports)")
    return trimmed


def fetch_weather_batch(
    airports_batch: pd.DataFrame, start_date: str, end_date: str
) -> None:
    lats = ",".join(str(v) for v in airports_batch["lat"])
    lons = ",".join(str(v) for v in airports_batch["lon"])
    params = {
        "latitude": lats,
        "longitude": lons,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_HOURLY_VARS),
        "timezone": "auto",
    }
    for attempt in range(MAX_RETRIES):
        resp = requests.get(ARCHIVE_API, params=params, headers=HTTP_HEADERS, timeout=120)
        if resp.status_code == 429:
            retry_after = float(resp.headers.get("Retry-After", 0)) or min(10 * (2**attempt), 300)
            print(f"  429 rate-limited, backing off {retry_after:.0f}s (attempt {attempt + 1}/{MAX_RETRIES})...")
            time.sleep(retry_after)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Gave up after {MAX_RETRIES} retries on 429 rate limiting")

    payload = resp.json()

    # Single-location requests return one object; multi-location return a list.
    results = payload if isinstance(payload, list) else [payload]

    for iata, result in zip(airports_batch["iata"], results):
        out_path = WEATHER_RAW_DIR / f"{iata}.parquet"
        hourly = result["hourly"]
        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        WEATHER_RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} hourly rows)")


def download_weather(start_date: str, end_date: str, force: bool = False) -> None:
    iata_codes = collect_iata_codes()
    airports = build_airports_csv(iata_codes)

    todo = airports if force else airports[
        ~airports["iata"].apply(lambda c: (WEATHER_RAW_DIR / f"{c}.parquet").exists())
    ]
    print(f"fetching weather for {len(todo)} airports ({len(airports) - len(todo)} cached)...")

    for i in range(0, len(todo), BATCH_SIZE):
        batch = todo.iloc[i : i + BATCH_SIZE]
        print(f"batch {i // BATCH_SIZE + 1}: {list(batch['iata'])}")
        fetch_weather_batch(batch, start_date, end_date)
        time.sleep(BATCH_SLEEP_SECONDS)  # be polite to the free API


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    download_weather(args.start_date, args.end_date, force=args.force)
