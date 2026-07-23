"""Evaluate the trained bucketed delay-severity model on the held-out
(chronologically final) test months: accuracy, log-loss, per-class
precision/recall, a confusion matrix, "adjacent-bucket" accuracy (since the
5 buckets are ordinal — predicting "late" when it was actually "very_late"
is a much smaller miss than predicting "on_time"), and a per-class
calibration check.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss,
)

from common import MODELS_DIR
from train import BUCKET_LABELS, CATEGORICAL_COLS, TARGET_COL, apply_calibration, apply_category_maps


def load_model_and_metadata():
    booster = xgb.Booster()
    booster.load_model(str(MODELS_DIR / "model.json"))
    with open(MODELS_DIR / "feature_metadata.json") as f:
        metadata = json.load(f)
    return booster, metadata


def main():
    test = pd.read_parquet(MODELS_DIR / "_test_split.parquet")
    booster, metadata = load_model_and_metadata()

    test = apply_category_maps(test, metadata["category_maps"])
    feature_cols = metadata["feature_order"]
    X_test = test[feature_cols]
    y_test = test[TARGET_COL].astype(int).to_numpy()

    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    raw_probs = booster.predict(dtest)  # (n, 5)
    probs = apply_calibration(raw_probs, metadata.get("calibration"))
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    ll = log_loss(y_test, probs, labels=list(range(len(BUCKET_LABELS))))
    print(f"Accuracy: {acc:.4f}  Log-loss: {ll:.4f}")

    adjacent_ok = (np.abs(preds - y_test) <= 1).mean()
    print(f"Adjacent-bucket accuracy (predicted within 1 bucket of actual): {adjacent_ok:.4f}")

    print("\nPer-class report:")
    print(classification_report(y_test, preds, target_names=BUCKET_LABELS, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, preds, labels=list(range(len(BUCKET_LABELS))))
    header = "        " + "".join(f"{label[:8]:>10}" for label in BUCKET_LABELS)
    print(header)
    for i, label in enumerate(BUCKET_LABELS):
        row = "".join(f"{cm[i, j]:>10,}" for j in range(len(BUCKET_LABELS)))
        print(f"{label[:8]:>8}{row}")

    print("\nPer-class calibration (predicted vs actual rate, by decile of that class's probability):")
    for k, label in enumerate(BUCKET_LABELS):
        y_binary = (y_test == k).astype(int)
        p_k = probs[:, k]
        try:
            deciles = pd.qcut(p_k, 10, duplicates="drop")
        except ValueError:
            print(f"  {label}: not enough distinct probability values to bin")
            continue
        grouped = pd.DataFrame({"p": p_k, "y": y_binary, "bin": deciles}).groupby("bin", observed=True)
        gap = float((grouped["p"].mean() - grouped["y"].mean()).abs().max())
        print(f"  {label}: max calibration gap = {gap:.3f}")


if __name__ == "__main__":
    main()
