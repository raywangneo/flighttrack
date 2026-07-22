"""Copy the trained model artifacts from ml/models/ (and the derived
airports/climatology reference files) into backend/app/model/, where the
FastAPI backend loads them from. Run after train.py + build_features.py
have produced fresh artifacts.
"""

import shutil

from common import AIRPORTS_CSV, CLIMATOLOGY_PATH, ML_ROOT, MODELS_DIR

BACKEND_MODEL_DIR = ML_ROOT.parent / "backend" / "app" / "model"

FILES_TO_COPY = [
    (MODELS_DIR / "model.json", BACKEND_MODEL_DIR / "model.json"),
    (MODELS_DIR / "feature_metadata.json", BACKEND_MODEL_DIR / "feature_metadata.json"),
    (AIRPORTS_CSV, BACKEND_MODEL_DIR / "airports.csv"),
    (CLIMATOLOGY_PATH, BACKEND_MODEL_DIR / "weather_climatology.parquet"),
]

if __name__ == "__main__":
    BACKEND_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for src, dst in FILES_TO_COPY:
        if not src.exists():
            raise FileNotFoundError(f"Missing expected artifact: {src}")
        shutil.copy2(src, dst)
        print(f"copied {src} -> {dst}")
