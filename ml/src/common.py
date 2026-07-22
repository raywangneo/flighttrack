"""Shared constants for the FlightTrack ml pipeline."""

from pathlib import Path

ML_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ML_ROOT / "data" / "raw"
WEATHER_RAW_DIR = RAW_DIR / "weather"
PROCESSED_DIR = ML_ROOT / "data" / "processed"
MODELS_DIR = ML_ROOT / "models"
AIRPORTS_CSV = ML_ROOT / "data" / "airports.csv"
CLIMATOLOGY_PATH = ML_ROOT / "data" / "weather_climatology.parquet"

# Raw BTS column name for the airline that actually operated the flight
# (more predictive of delay patterns than the marketing/codeshare brand).
# Confirmed against a live download of the Marketing_Carrier export — this
# dataset has no "Reporting_Airline" column at all, unlike the legacy export.
AIRLINE_SOURCE_COL = "IATA_Code_Operating_Airline"

BTS_COLUMNS = [
    "FlightDate",
    AIRLINE_SOURCE_COL,
    "Origin",
    "Dest",
    "CRSDepTime",
    "DepDelayMinutes",
    "CRSArrTime",
    "ArrDelayMinutes",
    "ArrDel15",
    "Cancelled",
    "Diverted",
    "Distance",
    "DayOfWeek",
    "Month",
    "Year",
    "CRSElapsedTime",
]

# Hourly variables confirmed available from Open-Meteo's historical archive API.
# No visibility field exists in this API; cloud_cover_low + precipitation/snowfall
# are used as a proxy instead.
WEATHER_HOURLY_VARS = [
    "precipitation",
    "rain",
    "snowfall",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "cloud_cover_low",
]

HTTP_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}
