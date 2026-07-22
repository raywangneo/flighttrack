"""Download BTS On-Time Performance monthly data and reduce to a per-month
Parquet file with only the columns FlightTrack needs.

BTS publishes two parallel monthly exports of the same underlying flight
data: "Marketing_Carrier" (includes codeshare/marketing-brand columns) and
"Reporting_Carrier" (the legacy, smaller export). Both were confirmed live
for months spanning 2019-2026, both contain the Reporting_Airline column
this pipeline actually uses, and either is fine as a source — the code
tries the modern name first and falls back to the legacy one only if a
request fails, treating a small/non-zip response as a miss.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from datetime import date

import pandas as pd
import requests

from common import AIRLINE_SOURCE_COL, BTS_COLUMNS, HTTP_HEADERS, PROCESSED_DIR, RAW_DIR

URL_PATTERNS = [
    "https://transtats.bts.gov/PREZIP/On_Time_Marketing_Carrier_On_Time_Performance_Beginning_January_2018_{year}_{month}.zip",
    "https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip",
]

# A real BTS monthly zip is tens of MB; treat anything tiny as an error page.
MIN_VALID_BYTES = 1_000_000


def _fetch_month_zip(year: int, month: int) -> bytes:
    last_error = None
    for pattern in URL_PATTERNS:
        url = pattern.format(year=year, month=month)
        try:
            resp = requests.get(url, headers=HTTP_HEADERS, timeout=120)
        except requests.RequestException as exc:
            last_error = exc
            continue
        if resp.status_code == 200 and len(resp.content) >= MIN_VALID_BYTES:
            print(f"  fetched {url} ({len(resp.content) / 1e6:.1f} MB)")
            return resp.content
        last_error = RuntimeError(
            f"{url} -> status={resp.status_code} size={len(resp.content)}"
        )
    raise RuntimeError(f"No valid BTS zip found for {year}-{month:02d}: {last_error}")


def download_month(year: int, month: int, force: bool = False) -> None:
    out_path = PROCESSED_DIR / f"bts_{year}_{month:02d}.parquet"
    if out_path.exists() and not force:
        print(f"skip {year}-{month:02d} (already processed)")
        return

    print(f"downloading {year}-{month:02d}...")
    raw_bytes = _fetch_month_zip(year, month)

    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csv_names:
            raise RuntimeError(f"No CSV found in zip for {year}-{month:02d}")
        with zf.open(csv_names[0]) as f:
            df = pd.read_csv(f, usecols=lambda c: c in BTS_COLUMNS, low_memory=False)

    df = df[[c for c in BTS_COLUMNS if c in df.columns]]
    df = df.rename(columns={AIRLINE_SOURCE_COL: "Reporting_Airline"})
    if "Reporting_Airline" not in df.columns:
        raise RuntimeError(
            f"Expected airline column '{AIRLINE_SOURCE_COL}' missing from {year}-{month:02d} "
            "download — BTS may have changed its schema again; inspect the raw CSV header."
        )
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"  wrote {out_path} ({len(df):,} rows)")


def month_range(start: date, end_inclusive: date):
    y, m = start.year, start.month
    while (y, m) <= (end_inclusive.year, end_inclusive.month):
        yield y, m
        m += 1
        if m > 12:
            m = 1
            y += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", required=True, help="YYYY-MM, e.g. 2024-06")
    parser.add_argument("--end", required=True, help="YYYY-MM, e.g. 2026-05")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    start_y, start_m = (int(x) for x in args.start.split("-"))
    end_y, end_m = (int(x) for x in args.end.split("-"))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    failed = []
    for y, m in month_range(date(start_y, start_m, 1), date(end_y, end_m, 1)):
        try:
            download_month(y, m, force=args.force)
        except RuntimeError as exc:
            print(f"  FAILED {y}-{m:02d}: {exc}")
            failed.append(f"{y}-{m:02d}")

    if failed:
        print(f"\n{len(failed)} month(s) unavailable (likely not yet finalized by BTS): {failed}")
