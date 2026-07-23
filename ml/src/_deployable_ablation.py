"""One-off: train a variant WITHOUT the upstream-delay features (prior_arr_delay,
scheduled_turnaround_minutes, first_flight_of_day) to measure the honest
deployable improvement from congestion + rolling-performance alone, without
the distribution-shift penalty of forcing constant defaults on a model that
was trained expecting those features to sometimes carry real signal.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.isotonic import IsotonicRegression

from common import PROCESSED_DIR
from train import (
    CATEGORICAL_COLS,
    TARGET_COL,
    apply_category_maps,
    build_category_maps,
    time_based_split,
)

DEPLOYABLE_NUMERIC_COLS = [
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

df = pd.read_parquet(PROCESSED_DIR / "features.parquet")
df["Reporting_Airline"] = df["Reporting_Airline"].astype(str)
df["Origin"] = df["Origin"].astype(str)
df["Dest"] = df["Dest"].astype(str)

train, val, test = time_based_split(df)
category_maps = build_category_maps(pd.concat([train, val, test]))

feature_cols = CATEGORICAL_COLS + DEPLOYABLE_NUMERIC_COLS
train_c = apply_category_maps(train, category_maps)
val_c = apply_category_maps(val, category_maps)
test_c = apply_category_maps(test, category_maps)

pos = (train_c[TARGET_COL] == 1).sum()
neg = (train_c[TARGET_COL] == 0).sum()
scale_pos_weight = neg / max(pos, 1)

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
model.fit(
    train_c[feature_cols], train_c[TARGET_COL],
    eval_set=[(val_c[feature_cols], val_c[TARGET_COL])],
    verbose=False,
)
print(f"best iteration: {model.best_iteration}")

# Calibrate on validation, same as train.py
raw_val_probs = model.predict_proba(val_c[feature_cols])[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(raw_val_probs, val_c[TARGET_COL])

raw_test_probs = model.predict_proba(test_c[feature_cols])[:, 1]
calibrated_test_probs = iso.predict(raw_test_probs)

auc = roc_auc_score(test_c[TARGET_COL], calibrated_test_probs)
pr_auc = average_precision_score(test_c[TARGET_COL], calibrated_test_probs)
print(f"Deployable-clean (no upstream-delay features at all):")
print(f"  ROC-AUC: {auc:.4f}")
print(f"  PR-AUC:  {pr_auc:.4f}")
