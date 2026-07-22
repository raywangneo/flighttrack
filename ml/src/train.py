"""Train a flight-delay classifier on ml/data/processed/features.parquet.

Uses a strictly time-based train/val/test split (never random) since flight
delay patterns are temporally correlated (seasonality, serially-correlated
weather events, schedule changes) — a random row-level split would let the
model see the same storm or route pattern in both train and test, inflating
validation metrics relative to how the model is actually used (predicting
forward from data already seen).

Fits a quick LogisticRegression baseline first as a pipeline sanity check,
then an XGBoost classifier (native categorical support, so Airline/Origin/
Dest don't need one-hot encoding). Serializes:
  - ml/models/model.json            (XGBoost native format, portable)
  - ml/models/feature_metadata.json (feature order, category maps, threshold)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from common import MODELS_DIR, PROCESSED_DIR

FEATURES_PATH = PROCESSED_DIR / "features.parquet"

CATEGORICAL_COLS = ["Reporting_Airline", "Origin", "Dest"]
NUMERIC_COLS = [
    "sched_dep_hour",
    "sched_arr_hour",
    "DayOfWeek",
    "Month",
    "Distance",
    "CRSElapsedTime",
    "origin_precipitation",
    "origin_rain",
    "origin_snowfall",
    "origin_wind_speed_10m",
    "origin_wind_gusts_10m",
    "origin_cloud_cover",
    "origin_cloud_cover_low",
    "dest_precipitation",
    "dest_rain",
    "dest_snowfall",
    "dest_wind_speed_10m",
    "dest_wind_gusts_10m",
    "dest_cloud_cover",
    "dest_cloud_cover_low",
]
TARGET_COL = "ArrDel15"


def time_based_split(df: pd.DataFrame):
    df = df.sort_values("FlightDate").reset_index(drop=True)
    months = df["FlightDate"].dt.to_period("M")
    unique_months = sorted(months.unique())
    if len(unique_months) < 6:
        raise RuntimeError(
            f"Only {len(unique_months)} distinct months in the data — need enough "
            "history for a meaningful time-based split. Download more months first."
        )
    n_val = max(1, round(len(unique_months) * 0.1))
    n_test = max(1, round(len(unique_months) * 0.1))
    train_months = unique_months[: -(n_val + n_test)]
    val_months = unique_months[-(n_val + n_test) : -n_test]
    test_months = unique_months[-n_test:]

    train = df[months.isin(train_months)]
    val = df[months.isin(val_months)]
    test = df[months.isin(test_months)]
    print(
        f"split: train={train_months[0]}..{train_months[-1]} ({len(train):,} rows), "
        f"val={val_months[0]}..{val_months[-1]} ({len(val):,} rows), "
        f"test={test_months[0]}..{test_months[-1]} ({len(test):,} rows)"
    )
    return train, val, test


def fit_baseline(train: pd.DataFrame, val: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    pipe = Pipeline(
        [("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))]
    )
    train_filled = train[NUMERIC_COLS].fillna(train[NUMERIC_COLS].median())
    val_filled = val[NUMERIC_COLS].fillna(train[NUMERIC_COLS].median())

    X_train = pd.concat([train[CATEGORICAL_COLS].reset_index(drop=True), train_filled.reset_index(drop=True)], axis=1)
    X_val = pd.concat([val[CATEGORICAL_COLS].reset_index(drop=True), val_filled.reset_index(drop=True)], axis=1)

    pipe.fit(X_train, train[TARGET_COL])
    preds = pipe.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(val[TARGET_COL], preds)
    print(f"baseline LogisticRegression val ROC-AUC: {auc:.4f}")
    return auc


def build_category_maps(df: pd.DataFrame) -> dict[str, list[str]]:
    return {col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in CATEGORICAL_COLS}


def apply_category_maps(df: pd.DataFrame, category_maps: dict[str, list[str]]) -> pd.DataFrame:
    df = df.copy()
    for col, categories in category_maps.items():
        df[col] = pd.Categorical(df[col].astype(str), categories=categories)
    return df


def fit_xgboost(train: pd.DataFrame, val: pd.DataFrame, category_maps: dict[str, list[str]]):
    train = apply_category_maps(train, category_maps)
    val = apply_category_maps(val, category_maps)

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS
    X_train, y_train = train[feature_cols], train[TARGET_COL]
    X_val, y_val = val[feature_cols], val[TARGET_COL]

    pos = (y_train == 1).sum()
    neg = (y_train == 0).sum()
    scale_pos_weight = neg / max(pos, 1)
    print(f"class balance: {pos:,} delayed / {neg:,} on-time (scale_pos_weight={scale_pos_weight:.2f})")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=5,
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        enable_categorical=True,
        eval_metric="auc",
        early_stopping_rounds=20,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    print(f"best iteration: {model.best_iteration}")
    return model, feature_cols


def fit_calibration(model, val: pd.DataFrame, category_maps: dict, feature_cols: list[str]) -> dict:
    """Fit an isotonic regression mapping raw XGBoost probabilities to
    calibrated ones, using the validation fold (not train, not test).
    Persisted as breakpoints so predict.py can reproduce it with a plain
    np.interp call — no need to ship a pickled sklearn object."""
    val = apply_category_maps(val, category_maps)
    raw_probs = model.predict_proba(val[feature_cols])[:, 1]

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_probs, val[TARGET_COL])

    gap = float(np.max(np.abs(iso.predict(raw_probs) - raw_probs)))
    print(f"calibration fit: max adjustment on validation set = {gap:.3f}")

    return {
        "method": "isotonic",
        "x_thresholds": iso.X_thresholds_.tolist(),
        "y_thresholds": iso.y_thresholds_.tolist(),
    }


def apply_calibration(raw_probs: np.ndarray, calibration: dict) -> np.ndarray:
    if not calibration:
        return raw_probs
    return np.interp(raw_probs, calibration["x_thresholds"], calibration["y_thresholds"])


def main():
    print(f"loading {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str)
    df["Origin"] = df["Origin"].astype(str)
    df["Dest"] = df["Dest"].astype(str)

    train, val, test = time_based_split(df)

    fit_baseline(train, val)

    category_maps = build_category_maps(pd.concat([train, val, test]))
    model, feature_cols = fit_xgboost(train, val, category_maps)

    numeric_medians = {col: float(train[col].median()) for col in NUMERIC_COLS}

    calibration = fit_calibration(model, val, category_maps, feature_cols)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(MODELS_DIR / "model.json"))

    metadata = {
        "feature_order": feature_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "category_maps": category_maps,
        "numeric_medians": numeric_medians,
        "target_col": TARGET_COL,
        "decision_threshold": 0.5,
        "calibration": calibration,
    }
    with open(MODELS_DIR / "feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved model to {MODELS_DIR / 'model.json'}")
    print(f"saved metadata to {MODELS_DIR / 'feature_metadata.json'}")

    # Stash test set split boundaries for evaluate.py
    test.to_parquet(MODELS_DIR / "_test_split.parquet", index=False)


if __name__ == "__main__":
    main()
