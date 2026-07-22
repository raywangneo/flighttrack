from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .airports import list_airports, load_airports
from .predict import get_supported_airlines, predict as run_predict, warm_up
from .schemas import PredictRequest, PredictResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    warm_up()
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


def is_valid_iata(code: str) -> bool:
    return code.upper() in set(load_airports()["iata"])
