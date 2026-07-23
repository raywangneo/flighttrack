"""Train a flight-delay-severity classifier on ml/data/processed/features.parquet.

Predicts which of 5 delay-severity buckets a flight will land in, rather
than a single binary "delayed >=15min" flag:
  on_time (<15min), little_late (15-30), late (30-60), very_late (60-120),
  mega_late (>120min). Early arrivals are treated as on_time.

Feature set is the "deployable-clean" set validated earlier this session:
schedule + weather + congestion + rolling recent performance. Upstream
aircraft-delay features (prior_arr_delay, scheduled_turnaround_minutes,
first_flight_of_day) were deliberately dropped — an ablation showed they
give a large offline accuracy boost but are undeployable (a future flight's
aircraft assignment is unknowable in advance), and training on them while
always imputing constant defaults at serving time actually performed worse
than not having them at all (distribution shift from the imputed constant).

Uses a strictly time-based train/val/test split (never random) since flight
delay patterns are temporally correlated — a random row-level split would
let the model see the same storm or route pattern in both train and test,
inflating validation metrics relative to how the model is actually used.

Fits a quick LogisticRegression baseline first as a pipeline sanity check,
then an XGBoost multi-class classifier (native categorical support, so
Airline/Origin/Dest don't need one-hot encoding). Serializes:
  - ml/models/model.json            (XGBoost native format, portable)
  - ml/models/feature_metadata.json (feature order, category maps, calibration)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

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
    "origin_hourly_traffic",
    "dest_hourly_traffic",
    "rolling_ontime_rate",
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
TARGET_COL = "delay_bucket"

BUCKET_EDGES = [15, 30, 60, 120]  # minutes
BUCKET_LABELS = ["on_time", "little_late", "late", "very_late", "mega_late"]


def derive_bucket(arr_delay_minutes: pd.Series) -> pd.Series:
    delay = arr_delay_minutes.clip(lower=0)  # early arrivals count as on_time
    bins = [-0.01] + BUCKET_EDGES + [float("inf")]
    return pd.cut(delay, bins=bins, labels=range(len(BUCKET_LABELS))).astype(int)


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
    from sklearn.metrics import accuracy_score, log_loss

    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
            ("num", StandardScaler(), NUMERIC_COLS),
        ]
    )
    pipe = Pipeline([("prep", preprocessor), ("clf", LogisticRegression(max_iter=1000))])
    train_filled = train[NUMERIC_COLS].fillna(train[NUMERIC_COLS].median())
    val_filled = val[NUMERIC_COLS].fillna(train[NUMERIC_COLS].median())

    X_train = pd.concat([train[CATEGORICAL_COLS].reset_index(drop=True), train_filled.reset_index(drop=True)], axis=1)
    X_val = pd.concat([val[CATEGORICAL_COLS].reset_index(drop=True), val_filled.reset_index(drop=True)], axis=1)

    pipe.fit(X_train, train[TARGET_COL])
    acc = accuracy_score(val[TARGET_COL], pipe.predict(X_val))
    ll = log_loss(val[TARGET_COL], pipe.predict_proba(X_val), labels=list(range(len(BUCKET_LABELS))))
    print(f"baseline LogisticRegression val accuracy: {acc:.4f}, log-loss: {ll:.4f}")
    return acc


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

    print("class balance:", y_train.value_counts().sort_index().to_dict())
    sample_weight = compute_sample_weight("balanced", y_train)
    # Early stopping picks the round that minimizes eval_set loss. If that
    # loss is unweighted while training is weighted, early stopping judges
    # against a different objective than the one being optimized — on an
    # 80%-majority-class validation set, unweighted mlogloss heavily favors
    # collapsing to the majority class, silently undoing the balanced
    # training. Weight the validation set the same way so both agree.
    sample_weight_val = compute_sample_weight("balanced", y_val)

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(BUCKET_LABELS),
        max_depth=6,
        n_estimators=500,
        learning_rate=0.05,
        enable_categorical=True,
        eval_metric="mlogloss",
        early_stopping_rounds=20,
    )
    model.fit(
        X_train, y_train, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], sample_weight_eval_set=[sample_weight_val],
        verbose=False,
    )
    print(f"best iteration: {model.best_iteration}")
    return model, feature_cols


# NOTE on multi-class calibration: an earlier version of this file fit one
# isotonic regression per class (one-vs-rest) and renormalized to sum to 1,
# mirroring the binary model's calibration approach. That produced a badly
# broken model in practice: with on_time at ~80% base rate, its isotonic
# curve stays elevated (up to 1.0) across almost the whole raw-score range,
# while minority classes' curves never exceed ~0.15-0.65 even at their own
# highest raw scores — so after renormalizing, on_time swamped every
# prediction regardless of what the raw model actually ranked highest
# (verified directly: a true mega_late test row with raw probs favoring
# mega_late [0.25] over on_time [0.07] came out of calibration as 60%
# on_time). Per-class OvR calibration doesn't account for competing
# probability mass from other classes, and breaks down badly under class
# imbalance this severe. Raw softmax probabilities are used directly
# instead — verified to give meaningfully non-zero precision/recall across
# all 5 classes, unlike the calibrated version's near-total collapse to
# on_time. Revisit with proper temperature scaling (a single scalar over
# the logits, which preserves relative ranking) if better-calibrated
# absolute probabilities are needed later.


def apply_calibration(raw_probs: np.ndarray, calibration: dict | None) -> np.ndarray:
    """Currently always a passthrough (calibration is None) — kept as the
    integration point so predict.py doesn't need to change if a working
    multi-class calibration method is added later."""
    if not calibration:
        return raw_probs
    single_row = raw_probs.ndim == 1
    probs = raw_probs.reshape(1, -1) if single_row else raw_probs

    calibrated = np.zeros_like(probs, dtype=float)
    for k, cal in enumerate(calibration["per_class"]):
        calibrated[:, k] = np.interp(probs[:, k], cal["x_thresholds"], cal["y_thresholds"])

    row_sums = np.clip(calibrated.sum(axis=1, keepdims=True), 1e-9, None)
    calibrated = calibrated / row_sums
    return calibrated[0] if single_row else calibrated


def main():
    print(f"loading {FEATURES_PATH}...")
    df = pd.read_parquet(FEATURES_PATH)
    df["Reporting_Airline"] = df["Reporting_Airline"].astype(str)
    df["Origin"] = df["Origin"].astype(str)
    df["Dest"] = df["Dest"].astype(str)
    df[TARGET_COL] = derive_bucket(df["ArrDelayMinutes"])
    print("overall bucket distribution:", df[TARGET_COL].value_counts().sort_index().to_dict())

    train, val, test = time_based_split(df)

    fit_baseline(train, val)

    category_maps = build_category_maps(pd.concat([train, val, test]))
    model, feature_cols = fit_xgboost(train, val, category_maps)

    numeric_medians = {col: float(train[col].median()) for col in NUMERIC_COLS}

    # See the NOTE above apply_calibration() — per-class isotonic calibration
    # was tried and found to badly break multi-class predictions under this
    # much class imbalance. Raw softmax probabilities are used directly.
    calibration = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.get_booster().save_model(str(MODELS_DIR / "model.json"))

    metadata = {
        "feature_order": feature_cols,
        "categorical_cols": CATEGORICAL_COLS,
        "numeric_cols": NUMERIC_COLS,
        "category_maps": category_maps,
        "numeric_medians": numeric_medians,
        "target_col": TARGET_COL,
        "bucket_labels": BUCKET_LABELS,
        "bucket_edges_minutes": BUCKET_EDGES,
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
