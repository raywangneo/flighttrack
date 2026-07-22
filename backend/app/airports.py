from functools import lru_cache
from pathlib import Path

import pandas as pd

AIRPORTS_CSV = Path(__file__).resolve().parent / "model" / "airports.csv"


@lru_cache
def load_airports() -> pd.DataFrame:
    return pd.read_csv(AIRPORTS_CSV)


def is_valid_airport(iata: str) -> bool:
    return iata.upper() in set(load_airports()["iata"])


def get_lat_lon(iata: str) -> tuple[float, float]:
    row = load_airports().set_index("iata").loc[iata.upper()]
    return float(row["lat"]), float(row["lon"])


def list_airports() -> list[str]:
    return sorted(load_airports()["iata"].tolist())
