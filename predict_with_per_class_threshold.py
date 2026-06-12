#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
predict_with_per_class_threshold.py

Use per-class thresholds for multi-label ECG prediction.

Inputs:
    model_output/ecg_1d_cnn_best.keras
    output_split/X_test.npy
    output_split/label_map.csv
    threshold_tuning_output/best_thresholds_per_class.npy

Outputs:
    prediction_output/Y_pred_prob.npy
    prediction_output/Y_pred_binary_per_class.npy
    prediction_output/predicted_dx_codes.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf


def normalize_per_sample_per_lead(X):
    """
    Normalize each ECG sample and each lead independently along the time axis.

    X shape:
        (N, 5000, 12)
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mean) / std).astype(np.float32)


def main():
    # =========================
    # 1. Set paths
    # =========================
    model_path = Path("model_output/ecg_1d_cnn_best.keras")
    x_path = Path("output_split/X_test.npy")
    label_map_path = Path("output_split/label_map.csv")
    thresholds_path = Path("threshold_tuning_output/best_thresholds_per_class.npy")

    output_dir = Path("prediction_output")
    output_dir.mkdir(exist_ok=True)

    # =========================
    # 2. Load files
    # =========================
    print("========== Loading files ==========")

    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model: {model_path}")

    if not x_path.exists():
        raise FileNotFoundError(f"Cannot find X_test: {x_path}")

    if not label_map_path.exists():
        raise FileNotFoundError(f"Cannot find label_map: {label_map_path}")

    if not thresholds_path.exists():
        raise FileNotFoundError(f"Cannot find thresholds: {thresholds_path}")

    model = tf.keras.models.load_model(model_path)
    X = np.load(x_path)
    label_map = pd.read_csv(label_map_path)
    thresholds = np.load(thresholds_path).astype(np.float32)

    print("X shape:", X.shape)
    print("thresholds shape:", thresholds.shape)
    print("label_map shape:", label_map.shape)

    if X.ndim != 3:
        raise ValueError(f"X should have shape (N, 5000, 12), got {X.shape}")

    if thresholds.shape[0] != len(label_map):
        raise ValueError(
            f"thresholds length and label_map length mismatch: "
            f"{thresholds.shape[0]} vs {len(label_map)}"
        )

    # =========================
    # 3. Normalize ECG
    # =========================
    print("\n========== Normalizing ECG ==========")

    X = normalize_per_sample_per_lead(X)

    print("X mean:", float(X.mean()))
    print("X std:", float(X.std()))

    # =========================
    # 4. Predict probabilities
    # =========================
    print("\n========== Predicting probabilities ==========")

    Y_pred_prob = model.predict(X, batch_size=32)

    print("Y_pred_prob shape:", Y_pred_prob.shape)
    print("Y_pred_prob min:", float(Y_pred_prob.min()))
    print("Y_pred_prob max:", float(Y_pred_prob.max()))

    # =========================
    # 5. Apply per-class thresholds
    # =========================
    print("\n========== Applying per-class thresholds ==========")

    Y_pred_binary = (Y_pred_prob >= thresholds.reshape(1, -1)).astype(np.int8)

    print("Y_pred_binary shape:", Y_pred_binary.shape)
    print("Predicted positive labels:", int(Y_pred_binary.sum()))

    # =========================
    # 6. Convert predictions to Dx code list
    # =========================
    print("\n========== Converting predictions to Dx codes ==========")

    rows = []

    for sample_idx in range(Y_pred_binary.shape[0]):
        positive_indices = np.where(Y_pred_binary[sample_idx] == 1)[0]

        predicted_codes = []
        predicted_probs = []
        used_thresholds = []

        for label_idx in positive_indices:
            dx_code = str(label_map.loc[label_idx, "dx_code"])
            prob = float(Y_pred_prob[sample_idx, label_idx])
            th = float(thresholds[label_idx])

            predicted_codes.append(dx_code)
            predicted_probs.append(f"{prob:.4f}")
            used_thresholds.append(f"{th:.2f}")

        rows.append({
            "sample_index": sample_idx,
            "num_predicted_labels": len(positive_indices),
            "predicted_dx_codes": "|".join(predicted_codes),
            "predicted_probabilities": "|".join(predicted_probs),
            "used_thresholds": "|".join(used_thresholds),
        })

    prediction_df = pd.DataFrame(rows)

    # =========================
    # 7. Save outputs
    # =========================
    print("\n========== Saving outputs ==========")

    np.save(output_dir / "Y_pred_prob.npy", Y_pred_prob)
    np.save(output_dir / "Y_pred_binary_per_class.npy", Y_pred_binary)
    prediction_df.to_csv(output_dir / "predicted_dx_codes.csv", index=False)

    print("Saved:", output_dir / "Y_pred_prob.npy")
    print("Saved:", output_dir / "Y_pred_binary_per_class.npy")
    print("Saved:", output_dir / "predicted_dx_codes.csv")

    print("\n========== Preview ==========")
    print(prediction_df.head())

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()