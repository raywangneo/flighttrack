from typing import Literal

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    airline: str = Field(..., description="Operating airline IATA code, e.g. 'AA'")
    origin: str = Field(..., description="Origin airport IATA code, e.g. 'JFK'")
    dest: str = Field(..., description="Destination airport IATA code, e.g. 'LAX'")
    scheduled_departure: str = Field(
        ...,
        description=(
            "Scheduled departure as ISO8601 local time AT THE ORIGIN AIRPORT, "
            "no UTC offset, e.g. '2026-08-15T14:30:00'."
        ),
    )


class PredictResponse(BaseModel):
    delay_probability: float = Field(..., description="Probability of arrival delay >= 15 min")
    delayed_prediction: bool
    risk_level: Literal["low", "medium", "high"]
    weather_source: Literal["forecast", "historical_average"]
    model_version: str
    caveats: list[str] = []
