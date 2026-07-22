"""Evaluate the trained model on the held-out (chronologically final) test
months: ROC-AUC, PR-AUC, precision/recall at default and F1-optimal
thresholds, and a calibration curve — calibration matters because the app
displays a raw probability, not just a classification.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

from common import MODELS_DIR
from train import CATEGORICAL_COLS, NUMERIC_COLS, TARGET_COL, apply_calibration, apply_category_maps


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
    y_test = test[TARGET_COL].astype(int)

    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    raw_probs = booster.predict(dtest)
    probs = apply_calibration(raw_probs, metadata.get("calibration"))

    auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    print(f"ROC-AUC: {auc:.4f}  (0.5 = random, higher is better)")
    print(f"PR-AUC:  {pr_auc:.4f}  (baseline = positive class rate = {y_test.mean():.4f})")

    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1s = 2 * precisions * recalls / np.clip(precisions + recalls, 1e-9, None)
    best_idx = np.argmax(f1s[:-1])
    print(
        f"F1-optimal threshold: {thresholds[best_idx]:.3f} "
        f"(precision={precisions[best_idx]:.3f}, recall={recalls[best_idx]:.3f}, f1={f1s[best_idx]:.3f})"
    )

    default_preds = (probs >= 0.5).astype(int)
    default_precision = (
        (default_preds & y_test.values).sum() / max(default_preds.sum(), 1)
    )
    default_recall = (default_preds & y_test.values).sum() / max(y_test.sum(), 1)
    print(f"At threshold 0.5: precision={default_precision:.3f}, recall={default_recall:.3f}")

    frac_pos, mean_pred = calibration_curve(y_test, probs, n_bins=10, strategy="quantile")
    print("\nCalibration (predicted vs actual delay rate, by decile):")
    for p, a in zip(mean_pred, frac_pos):
        bar = "#" * int(a * 50)
        print(f"  predicted={p:.3f}  actual={a:.3f}  {bar}")

    max_calibration_gap = float(np.max(np.abs(frac_pos - mean_pred)))
    print(f"\nmax calibration gap: {max_calibration_gap:.3f}")
    if max_calibration_gap > 0.10:
        print(
            "  warning: calibration gap > 0.10 — consider wrapping with "
            "CalibratedClassifierCV before shipping raw probabilities to the app."
        )


if __name__ == "__main__":
    main()
