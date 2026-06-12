#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate_model.py

Evaluate a multi-label ECG classification model using saved test labels and model predictions.

Default input files:
    output_split/Y_test.npy
    output_split/label_map.csv
    model_output/test_predictions.npy

Default outputs:
    evaluation_output/per_class_metrics.csv
    evaluation_output/overall_metrics.txt
    evaluation_output/threshold_search.csv
    evaluation_output/best_thresholds_per_class.csv
    evaluation_output/y_pred_binary_global.npy
    evaluation_output/y_pred_binary_per_class.npy

Main functions:
1. Compute overall metrics at a global threshold.
2. Compute per-class AUC, AP, precision, recall, F1, specificity, support.
3. Search global threshold from 0.05 to 0.95.
4. Search best per-class thresholds by F1.
5. Save all evaluation results.

Usage:
    python evaluate_model.py
    python evaluate_model.py --threshold 0.5
    python evaluate_model.py --threshold 0.3
    python evaluate_model.py --output-dir evaluation_output
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    hamming_loss,
    multilabel_confusion_matrix,
    classification_report,
)


def safe_auc(y_true, y_score):
    """Compute ROC-AUC safely. Return NaN if a class has only one label value."""
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_ap(y_true, y_score):
    """Compute average precision safely."""
    try:
        if np.sum(y_true) == 0:
            return np.nan
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def binarize_predictions(y_score, threshold):
    """Convert probabilities to binary predictions using a global threshold."""
    return (y_score >= threshold).astype(np.int8)


def compute_overall_metrics(y_true, y_score, threshold):
    """Compute overall multi-label metrics for a given threshold."""
    y_pred = binarize_predictions(y_score, threshold)

    metrics = {}
    metrics["threshold"] = threshold

    # Ranking-based metrics
    try:
        metrics["micro_auc"] = roc_auc_score(y_true, y_score, average="micro")
    except Exception:
        metrics["micro_auc"] = np.nan

    try:
        metrics["macro_auc"] = roc_auc_score(y_true, y_score, average="macro")
    except Exception:
        metrics["macro_auc"] = np.nan

    try:
        metrics["micro_average_precision"] = average_precision_score(y_true, y_score, average="micro")
    except Exception:
        metrics["micro_average_precision"] = np.nan

    try:
        metrics["macro_average_precision"] = average_precision_score(y_true, y_score, average="macro")
    except Exception:
        metrics["macro_average_precision"] = np.nan

    # Threshold-based metrics
    metrics["micro_precision"] = precision_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["micro_recall"] = recall_score(y_true, y_pred, average="micro", zero_division=0)
    metrics["micro_f1"] = f1_score(y_true, y_pred, average="micro", zero_division=0)

    metrics["macro_precision"] = precision_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(y_true, y_pred, average="macro", zero_division=0)
    metrics["macro_f1"] = f1_score(y_true, y_pred, average="macro", zero_division=0)

    metrics["weighted_precision"] = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["weighted_recall"] = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics["weighted_f1"] = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    metrics["subset_accuracy_exact_match"] = accuracy_score(y_true, y_pred)
    metrics["binary_accuracy"] = float((y_true == y_pred).mean())
    metrics["hamming_loss"] = hamming_loss(y_true, y_pred)

    metrics["true_positive_labels"] = int(y_true.sum())
    metrics["predicted_positive_labels"] = int(y_pred.sum())
    metrics["samples"] = int(y_true.shape[0])
    metrics["classes"] = int(y_true.shape[1])

    return metrics, y_pred


def compute_per_class_metrics(y_true, y_score, y_pred, label_map):
    """Compute per-class metrics."""
    rows = []
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        ys = y_score[:, i]
        yp = y_pred[:, i]

        tn, fp, fn, tp = mcm[i].ravel()
        support = int(yt.sum())
        pred_pos = int(yp.sum())

        precision = precision_score(yt, yp, zero_division=0)
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        row = {
            "label_index": i,
            "dx_code": str(label_map.loc[i, "dx_code"]) if "dx_code" in label_map.columns else str(i),
            "support": support,
            "predicted_positive": pred_pos,
            "auc": safe_auc(yt, ys),
            "average_precision": safe_ap(yt, ys),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "specificity": specificity,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }

        if "count" in label_map.columns:
            row["original_count"] = int(label_map.loc[i, "count"])
        if "filtered_count" in label_map.columns:
            row["filtered_count"] = int(label_map.loc[i, "filtered_count"])

        rows.append(row)

    return pd.DataFrame(rows)


def threshold_search(y_true, y_score, thresholds):
    """Search global thresholds and compute overall metrics for each threshold."""
    rows = []
    for th in thresholds:
        metrics, _ = compute_overall_metrics(y_true, y_score, float(th))
        rows.append(metrics)
    return pd.DataFrame(rows)


def find_best_thresholds_per_class(y_true, y_score, thresholds, label_map):
    """Find best threshold for each class based on F1 score."""
    rows = []
    best_binary = np.zeros_like(y_true, dtype=np.int8)

    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        ys = y_score[:, i]

        best = {
            "label_index": i,
            "dx_code": str(label_map.loc[i, "dx_code"]) if "dx_code" in label_map.columns else str(i),
            "best_threshold": 0.5,
            "best_f1": -1.0,
            "precision_at_best": 0.0,
            "recall_at_best": 0.0,
            "support": int(yt.sum()),
        }

        for th in thresholds:
            yp = (ys >= th).astype(np.int8)
            f1 = f1_score(yt, yp, zero_division=0)
            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)

            if f1 > best["best_f1"]:
                best["best_threshold"] = float(th)
                best["best_f1"] = float(f1)
                best["precision_at_best"] = float(precision)
                best["recall_at_best"] = float(recall)

        best_binary[:, i] = (ys >= best["best_threshold"]).astype(np.int8)
        rows.append(best)

    return pd.DataFrame(rows), best_binary


def write_overall_metrics(path, metrics, classification_report_text=None):
    """Write overall metrics to a text file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== Overall Multi-label Evaluation ==========\n\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

        if classification_report_text is not None:
            f.write("\n========== Classification Report ==========\n\n")
            f.write(classification_report_text)
            f.write("\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-label ECG classification predictions.")

    parser.add_argument("--y-test", default="output_split/Y_test.npy", help="Path to Y_test.npy")
    parser.add_argument("--y-pred", default="model_output/test_predictions.npy", help="Path to test_predictions.npy")
    parser.add_argument("--label-map", default="output_split/label_map.csv", help="Path to label_map.csv")
    parser.add_argument("--output-dir", default="evaluation_output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Global decision threshold")
    parser.add_argument("--threshold-min", type=float, default=0.05, help="Minimum threshold for search")
    parser.add_argument("--threshold-max", type=float, default=0.95, help="Maximum threshold for search")
    parser.add_argument("--threshold-step", type=float, default=0.05, help="Threshold step for search")

    args = parser.parse_args()

    y_test_path = Path(args.y_test)
    y_pred_path = Path(args.y_pred)
    label_map_path = Path(args.label_map)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("========== Loading files ==========")
    print("Y_test:", y_test_path)
    print("Y_pred:", y_pred_path)
    print("label_map:", label_map_path)

    if not y_test_path.exists():
        raise FileNotFoundError(f"Cannot find Y_test file: {y_test_path}")
    if not y_pred_path.exists():
        raise FileNotFoundError(f"Cannot find prediction file: {y_pred_path}")
    if not label_map_path.exists():
        raise FileNotFoundError(f"Cannot find label_map file: {label_map_path}")

    Y_test = np.load(y_test_path).astype(np.int8)
    Y_pred = np.load(y_pred_path).astype(np.float32)
    label_map = pd.read_csv(label_map_path)

    print("Y_test shape:", Y_test.shape)
    print("Y_pred shape:", Y_pred.shape)
    print("label_map shape:", label_map.shape)
    print("Y_pred min/max:", float(Y_pred.min()), float(Y_pred.max()))

    if Y_test.shape != Y_pred.shape:
        raise ValueError(f"Shape mismatch: Y_test={Y_test.shape}, Y_pred={Y_pred.shape}")

    # =========================
    # Global threshold evaluation
    # =========================
    print("\n========== Global threshold evaluation ==========")
    overall_metrics, Y_binary_global = compute_overall_metrics(Y_test, Y_pred, args.threshold)
    per_class_df = compute_per_class_metrics(Y_test, Y_pred, Y_binary_global, label_map)

    target_names = label_map["dx_code"].astype(str).tolist() if "dx_code" in label_map.columns else None
    report_text = classification_report(
        Y_test,
        Y_binary_global,
        target_names=target_names,
        zero_division=0,
    )

    print("Threshold:", args.threshold)
    for key in [
        "micro_auc",
        "macro_auc",
        "micro_average_precision",
        "macro_average_precision",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "binary_accuracy",
        "hamming_loss",
    ]:
        print(f"{key}: {overall_metrics[key]}")

    per_class_path = output_dir / "per_class_metrics.csv"
    overall_path = output_dir / "overall_metrics.txt"
    y_binary_global_path = output_dir / "y_pred_binary_global.npy"

    per_class_df.to_csv(per_class_path, index=False)
    write_overall_metrics(overall_path, overall_metrics, report_text)
    np.save(y_binary_global_path, Y_binary_global)

    print("Saved:", per_class_path)
    print("Saved:", overall_path)
    print("Saved:", y_binary_global_path)

    # =========================
    # Threshold search
    # =========================
    print("\n========== Threshold search ==========")
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-9, args.threshold_step)
    thresholds = np.round(thresholds, 4)

    threshold_df = threshold_search(Y_test, Y_pred, thresholds)
    threshold_path = output_dir / "threshold_search.csv"
    threshold_df.to_csv(threshold_path, index=False)

    best_micro_idx = threshold_df["micro_f1"].idxmax()
    best_macro_idx = threshold_df["macro_f1"].idxmax()

    print("Best global threshold by micro-F1:")
    print(threshold_df.loc[best_micro_idx])
    print("\nBest global threshold by macro-F1:")
    print(threshold_df.loc[best_macro_idx])
    print("Saved:", threshold_path)

    # =========================
    # Per-class threshold optimization
    # =========================
    print("\n========== Per-class threshold optimization ==========")
    best_thresholds_df, Y_binary_per_class = find_best_thresholds_per_class(
        Y_test,
        Y_pred,
        thresholds,
        label_map,
    )

    best_thresholds_path = output_dir / "best_thresholds_per_class.csv"
    y_binary_per_class_path = output_dir / "y_pred_binary_per_class.npy"
    best_thresholds_df.to_csv(best_thresholds_path, index=False)
    np.save(y_binary_per_class_path, Y_binary_per_class)

    per_class_threshold_metrics = compute_overall_metrics(Y_test, Y_pred, args.threshold)[0]

    # Replace threshold-based metrics with per-class binary predictions
    per_class_threshold_metrics["threshold"] = "per-class optimized"
    per_class_threshold_metrics["micro_precision"] = precision_score(Y_test, Y_binary_per_class, average="micro", zero_division=0)
    per_class_threshold_metrics["micro_recall"] = recall_score(Y_test, Y_binary_per_class, average="micro", zero_division=0)
    per_class_threshold_metrics["micro_f1"] = f1_score(Y_test, Y_binary_per_class, average="micro", zero_division=0)
    per_class_threshold_metrics["macro_precision"] = precision_score(Y_test, Y_binary_per_class, average="macro", zero_division=0)
    per_class_threshold_metrics["macro_recall"] = recall_score(Y_test, Y_binary_per_class, average="macro", zero_division=0)
    per_class_threshold_metrics["macro_f1"] = f1_score(Y_test, Y_binary_per_class, average="macro", zero_division=0)
    per_class_threshold_metrics["weighted_precision"] = precision_score(Y_test, Y_binary_per_class, average="weighted", zero_division=0)
    per_class_threshold_metrics["weighted_recall"] = recall_score(Y_test, Y_binary_per_class, average="weighted", zero_division=0)
    per_class_threshold_metrics["weighted_f1"] = f1_score(Y_test, Y_binary_per_class, average="weighted", zero_division=0)
    per_class_threshold_metrics["subset_accuracy_exact_match"] = accuracy_score(Y_test, Y_binary_per_class)
    per_class_threshold_metrics["binary_accuracy"] = float((Y_test == Y_binary_per_class).mean())
    per_class_threshold_metrics["hamming_loss"] = hamming_loss(Y_test, Y_binary_per_class)
    per_class_threshold_metrics["predicted_positive_labels"] = int(Y_binary_per_class.sum())

    per_class_opt_path = output_dir / "overall_metrics_per_class_thresholds.txt"
    write_overall_metrics(per_class_opt_path, per_class_threshold_metrics)

    print("Saved:", best_thresholds_path)
    print("Saved:", y_binary_per_class_path)
    print("Saved:", per_class_opt_path)

    print("\nPer-class optimized threshold summary:")
    print("micro_precision:", per_class_threshold_metrics["micro_precision"])
    print("micro_recall:", per_class_threshold_metrics["micro_recall"])
    print("micro_f1:", per_class_threshold_metrics["micro_f1"])
    print("macro_f1:", per_class_threshold_metrics["macro_f1"])

    print("\n========== Top 10 classes by F1 at global threshold ==========")
    print(per_class_df.sort_values("f1", ascending=False).head(10)[[
        "label_index", "dx_code", "support", "auc", "precision", "recall", "f1"
    ]])

    print("\n========== Bottom 10 classes by F1 at global threshold ==========")
    print(per_class_df.sort_values("f1", ascending=True).head(10)[[
        "label_index", "dx_code", "support", "auc", "precision", "recall", "f1"
    ]])

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()