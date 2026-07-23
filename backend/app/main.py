from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .airports import list_airports, load_airports
from .predict import get_supported_airlines, predict as run_predict, warm_up
from .schedule import load_schedule_reference, search_by_flight_number, search_by_route
from .schemas import (
    FlightNumberSearchResponse,
    PredictRequest,
    PredictResponse,
    RouteSearchResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
    load_schedule_reference()
    yield


app = FastAPI(title="FlightTrack API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/airports")
def airports():
    return list_airports()


@app.get("/airlines")
def airlines():
    return get_supported_airlines()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not is_valid_iata(req.origin) or not is_valid_iata(req.dest):
        raise HTTPException(status_code=422, detail="Unknown origin or destination airport code")
    return run_predict(req)


@app.get("/schedule/by-flight-number", response_model=FlightNumberSearchResponse)
def schedule_by_flight_number(query: str):
    result = search_by_flight_number(query)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail="Flight number must look like 'DL1234' (2-letter airline code + 1-4 digit number)",
        )
    airline, number, itineraries = result
    return FlightNumberSearchResponse(airline=airline, flight_number=number, itineraries=itineraries)


@app.get("/schedule/by-route", response_model=RouteSearchResponse)
def schedule_by_route(origin: str, dest: str, time_of_day_minutes: int):
    if not is_valid_iata(origin) or not is_valid_iata(dest):
        raise HTTPException(status_code=422, detail="Unknown origin or destination airport code")
    if not 0 <= time_of_day_minutes < 24 * 60:
        raise HTTPException(status_code=422, detail="time_of_day_minutes must be in [0, 1440)")
    itineraries, band_hours = search_by_route(origin, dest, time_of_day_minutes)
    return RouteSearchResponse(itineraries=itineraries, band_hours=band_hours)


def is_valid_iata(code: str) -> bool:
    return code.upper() in set(load_airports()["iata"])
