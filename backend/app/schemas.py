from typing import Literal

from pydantic import BaseModel, Field

BucketName = Literal["on_time", "little_late", "late", "very_late", "mega_late"]


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


class BucketProbabilities(BaseModel):
    on_time: float = Field(..., description="Arrival delay < 15 min (or early)")
    little_late: float = Field(..., description="Arrival delay 15-30 min")
    late: float = Field(..., description="Arrival delay 30-60 min")
    very_late: float = Field(..., description="Arrival delay 60-120 min")
    mega_late: float = Field(..., description="Arrival delay > 120 min")


class PredictResponse(BaseModel):
    predicted_bucket: BucketName
    bucket_probability: float = Field(..., description="Calibrated probability of the predicted bucket")
    bucket_probabilities: BucketProbabilities
    weather_source: Literal["forecast", "historical_average"]
    model_version: str
    caveats: list[str] = []


class ItineraryCandidate(BaseModel):
    airline: str
    flight_number: str
    origin: str
    dest: str
    dep_time_minutes: int = Field(..., description="Scheduled departure, minutes since local midnight at origin")
    arr_time_minutes: int = Field(..., description="Scheduled arrival, minutes since local midnight at dest")
    distance_miles: float
    elapsed_minutes: float
    days_mask: int = Field(..., description="Bitmask of historically-operated weekdays: bit0=Mon .. bit6=Sun")
    sample_count: int = Field(..., description="How many historical flights informed this itinerary")


class FlightNumberSearchResponse(BaseModel):
    airline: str
    flight_number: str
    itineraries: list[ItineraryCandidate]


class RouteSearchResponse(BaseModel):
    itineraries: list[ItineraryCandidate]
    band_hours: int = Field(..., description="Actual search width that produced results: 1, 2, or 3")
