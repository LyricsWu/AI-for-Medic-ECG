#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tune_thresholds_on_val.py

Threshold tuning for multi-label ECG classification.

Important:
- Thresholds are selected on the validation set.
- Final metrics are reported on the test set.
- This avoids tuning thresholds directly on the test set.

Inputs:
    output_split/X_val.npy
    output_split/Y_val.npy
    output_split/X_test.npy
    output_split/Y_test.npy
    output_split/label_map.csv
    model_output/ecg_1d_cnn_best.keras

Outputs:
    threshold_tuning_output/val_predictions.npy
    threshold_tuning_output/test_predictions_best.npy
    threshold_tuning_output/val_threshold_search.csv
    threshold_tuning_output/best_thresholds_per_class.csv
    threshold_tuning_output/test_metrics_threshold_0_5.txt
    threshold_tuning_output/test_metrics_best_micro_threshold.txt
    threshold_tuning_output/test_metrics_best_macro_threshold.txt
    threshold_tuning_output/test_metrics_per_class_thresholds.txt
    threshold_tuning_output/per_class_test_metrics_0_5.csv
    threshold_tuning_output/per_class_test_metrics_best_micro.csv
    threshold_tuning_output/per_class_test_metrics_best_macro.csv
    threshold_tuning_output/per_class_test_metrics_per_class.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

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


def normalize_per_sample_per_lead(X):
    """
    Normalize each ECG sample and each lead independently along the time axis.

    X shape:
        (samples, time_points, leads)
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return ((X - mean) / std).astype(np.float32)


def safe_auc(y_true, y_score):
    try:
        if len(np.unique(y_true)) < 2:
            return np.nan
        return roc_auc_score(y_true, y_score)
    except Exception:
        return np.nan


def safe_average_precision(y_true, y_score):
    try:
        if np.sum(y_true) == 0:
            return np.nan
        return average_precision_score(y_true, y_score)
    except Exception:
        return np.nan


def binarize_global(Y_score, threshold):
    return (Y_score >= threshold).astype(np.int8)


def binarize_per_class(Y_score, thresholds):
    """
    thresholds shape:
        (num_classes,)
    """
    return (Y_score >= thresholds.reshape(1, -1)).astype(np.int8)


def compute_overall_metrics(Y_true, Y_score, Y_pred, threshold_name):
    metrics = {}

    metrics["threshold"] = threshold_name

    try:
        metrics["micro_auc"] = roc_auc_score(Y_true, Y_score, average="micro")
    except Exception:
        metrics["micro_auc"] = np.nan

    try:
        metrics["macro_auc"] = roc_auc_score(Y_true, Y_score, average="macro")
    except Exception:
        metrics["macro_auc"] = np.nan

    try:
        metrics["micro_average_precision"] = average_precision_score(Y_true, Y_score, average="micro")
    except Exception:
        metrics["micro_average_precision"] = np.nan

    try:
        metrics["macro_average_precision"] = average_precision_score(Y_true, Y_score, average="macro")
    except Exception:
        metrics["macro_average_precision"] = np.nan

    metrics["micro_precision"] = precision_score(Y_true, Y_pred, average="micro", zero_division=0)
    metrics["micro_recall"] = recall_score(Y_true, Y_pred, average="micro", zero_division=0)
    metrics["micro_f1"] = f1_score(Y_true, Y_pred, average="micro", zero_division=0)

    metrics["macro_precision"] = precision_score(Y_true, Y_pred, average="macro", zero_division=0)
    metrics["macro_recall"] = recall_score(Y_true, Y_pred, average="macro", zero_division=0)
    metrics["macro_f1"] = f1_score(Y_true, Y_pred, average="macro", zero_division=0)

    metrics["weighted_precision"] = precision_score(Y_true, Y_pred, average="weighted", zero_division=0)
    metrics["weighted_recall"] = recall_score(Y_true, Y_pred, average="weighted", zero_division=0)
    metrics["weighted_f1"] = f1_score(Y_true, Y_pred, average="weighted", zero_division=0)

    metrics["subset_accuracy_exact_match"] = accuracy_score(Y_true, Y_pred)
    metrics["binary_accuracy"] = float((Y_true == Y_pred).mean())
    metrics["hamming_loss"] = hamming_loss(Y_true, Y_pred)

    metrics["true_positive_labels"] = int(Y_true.sum())
    metrics["predicted_positive_labels"] = int(Y_pred.sum())
    metrics["samples"] = int(Y_true.shape[0])
    metrics["classes"] = int(Y_true.shape[1])

    return metrics


def compute_per_class_metrics(Y_true, Y_score, Y_pred, label_map):
    rows = []
    mcm = multilabel_confusion_matrix(Y_true, Y_pred)

    for i in range(Y_true.shape[1]):
        yt = Y_true[:, i]
        ys = Y_score[:, i]
        yp = Y_pred[:, i]

        tn, fp, fn, tp = mcm[i].ravel()

        precision = precision_score(yt, yp, zero_division=0)
        recall = recall_score(yt, yp, zero_division=0)
        f1 = f1_score(yt, yp, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

        row = {
            "label_index": i,
            "dx_code": str(label_map.loc[i, "dx_code"]),
            "support": int(yt.sum()),
            "predicted_positive": int(yp.sum()),
            "auc": safe_auc(yt, ys),
            "average_precision": safe_average_precision(yt, ys),
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


def write_metrics_txt(path, metrics, classification_report_text=None):
    with open(path, "w", encoding="utf-8") as f:
        f.write("========== Overall Metrics ==========\n\n")

        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

        if classification_report_text is not None:
            f.write("\n========== Classification Report ==========\n\n")
            f.write(classification_report_text)
            f.write("\n")


def threshold_search_on_validation(Y_val, P_val, thresholds):
    rows = []

    for th in thresholds:
        Y_pred = binarize_global(P_val, th)
        metrics = compute_overall_metrics(
            Y_true=Y_val,
            Y_score=P_val,
            Y_pred=Y_pred,
            threshold_name=float(th),
        )
        rows.append(metrics)

    return pd.DataFrame(rows)


def find_best_per_class_thresholds(Y_val, P_val, thresholds, label_map):
    rows = []
    best_thresholds = np.zeros(Y_val.shape[1], dtype=np.float32)

    for i in range(Y_val.shape[1]):
        yt = Y_val[:, i]
        ys = P_val[:, i]

        best_f1 = -1.0
        best_threshold = 0.5
        best_precision = 0.0
        best_recall = 0.0

        for th in thresholds:
            yp = (ys >= th).astype(np.int8)

            precision = precision_score(yt, yp, zero_division=0)
            recall = recall_score(yt, yp, zero_division=0)
            f1 = f1_score(yt, yp, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(th)
                best_precision = precision
                best_recall = recall

        best_thresholds[i] = best_threshold

        rows.append({
            "label_index": i,
            "dx_code": str(label_map.loc[i, "dx_code"]),
            "support_val": int(yt.sum()),
            "best_threshold": best_threshold,
            "best_val_f1": best_f1,
            "val_precision_at_best": best_precision,
            "val_recall_at_best": best_recall,
        })

    return pd.DataFrame(rows), best_thresholds


def evaluate_and_save(
    Y_true,
    P_score,
    Y_pred,
    label_map,
    threshold_name,
    output_txt_path,
    output_csv_path,
):
    metrics = compute_overall_metrics(
        Y_true=Y_true,
        Y_score=P_score,
        Y_pred=Y_pred,
        threshold_name=threshold_name,
    )

    target_names = label_map["dx_code"].astype(str).tolist()

    report_text = classification_report(
        Y_true,
        Y_pred,
        target_names=target_names,
        zero_division=0,
    )

    per_class_df = compute_per_class_metrics(
        Y_true=Y_true,
        Y_score=P_score,
        Y_pred=Y_pred,
        label_map=label_map,
    )

    write_metrics_txt(output_txt_path, metrics, report_text)
    per_class_df.to_csv(output_csv_path, index=False)

    return metrics, per_class_df


def print_metric_summary(name, metrics):
    print(f"\n========== {name} ==========")
    print("threshold:", metrics["threshold"])
    print("micro_auc:", metrics["micro_auc"])
    print("macro_auc:", metrics["macro_auc"])
    print("micro_precision:", metrics["micro_precision"])
    print("micro_recall:", metrics["micro_recall"])
    print("micro_f1:", metrics["micro_f1"])
    print("macro_precision:", metrics["macro_precision"])
    print("macro_recall:", metrics["macro_recall"])
    print("macro_f1:", metrics["macro_f1"])
    print("binary_accuracy:", metrics["binary_accuracy"])
    print("hamming_loss:", metrics["hamming_loss"])
    print("predicted_positive_labels:", metrics["predicted_positive_labels"])


def main():
    input_dir = Path("output_split")
    model_dir = Path("model_output")
    output_dir = Path("threshold_tuning_output")
    output_dir.mkdir(exist_ok=True)

    model_path = model_dir / "ecg_1d_cnn_best.keras"

    print("========== Loading data ==========")

    X_val = np.load(input_dir / "X_val.npy")
    Y_val = np.load(input_dir / "Y_val.npy").astype(np.int8)

    X_test = np.load(input_dir / "X_test.npy")
    Y_test = np.load(input_dir / "Y_test.npy").astype(np.int8)

    label_map = pd.read_csv(input_dir / "label_map.csv")

    print("X_val:", X_val.shape, X_val.dtype)
    print("Y_val:", Y_val.shape, Y_val.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("Y_test:", Y_test.shape, Y_test.dtype)
    print("label_map:", label_map.shape)

    print("\n========== Normalizing data ==========")

    X_val = normalize_per_sample_per_lead(X_val)
    X_test = normalize_per_sample_per_lead(X_test)

    print("X_val mean/std:", float(X_val.mean()), float(X_val.std()))
    print("X_test mean/std:", float(X_test.mean()), float(X_test.std()))

    print("\n========== Loading model ==========")
    print("model:", model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Cannot find model file: {model_path}")

    model = tf.keras.models.load_model(model_path)

    print("\n========== Predicting validation and test sets ==========")

    P_val = model.predict(X_val, batch_size=32)
    P_test = model.predict(X_test, batch_size=32)

    np.save(output_dir / "val_predictions.npy", P_val)
    np.save(output_dir / "test_predictions_best.npy", P_test)

    print("P_val:", P_val.shape, P_val.min(), P_val.max())
    print("P_test:", P_test.shape, P_test.min(), P_test.max())

    print("\n========== Threshold search on validation set ==========")

    thresholds = np.arange(0.05, 0.95 + 1e-9, 0.05)
    thresholds = np.round(thresholds, 4)

    val_threshold_df = threshold_search_on_validation(
        Y_val=Y_val,
        P_val=P_val,
        thresholds=thresholds,
    )

    val_threshold_df.to_csv(output_dir / "val_threshold_search.csv", index=False)

    best_micro_row = val_threshold_df.loc[val_threshold_df["micro_f1"].idxmax()]
    best_macro_row = val_threshold_df.loc[val_threshold_df["macro_f1"].idxmax()]

    best_micro_threshold = float(best_micro_row["threshold"])
    best_macro_threshold = float(best_macro_row["threshold"])

    print("\nBest validation global threshold by micro-F1:")
    print(best_micro_row)

    print("\nBest validation global threshold by macro-F1:")
    print(best_macro_row)

    print("\n========== Per-class threshold search on validation set ==========")

    best_thresholds_df, best_thresholds = find_best_per_class_thresholds(
        Y_val=Y_val,
        P_val=P_val,
        thresholds=thresholds,
        label_map=label_map,
    )

    best_thresholds_df.to_csv(output_dir / "best_thresholds_per_class.csv", index=False)
    np.save(output_dir / "best_thresholds_per_class.npy", best_thresholds)

    print(best_thresholds_df)

    print("\n========== Final test evaluation ==========")

    # 1. Default threshold = 0.5
    Y_test_pred_05 = binarize_global(P_test, 0.5)

    metrics_05, per_class_05 = evaluate_and_save(
        Y_true=Y_test,
        P_score=P_test,
        Y_pred=Y_test_pred_05,
        label_map=label_map,
        threshold_name="0.5",
        output_txt_path=output_dir / "test_metrics_threshold_0_5.txt",
        output_csv_path=output_dir / "per_class_test_metrics_0_5.csv",
    )

    # 2. Best global threshold by validation micro-F1
    Y_test_pred_best_micro = binarize_global(P_test, best_micro_threshold)

    metrics_best_micro, per_class_best_micro = evaluate_and_save(
        Y_true=Y_test,
        P_score=P_test,
        Y_pred=Y_test_pred_best_micro,
        label_map=label_map,
        threshold_name=f"best_val_micro_f1_threshold_{best_micro_threshold}",
        output_txt_path=output_dir / "test_metrics_best_micro_threshold.txt",
        output_csv_path=output_dir / "per_class_test_metrics_best_micro.csv",
    )

    # 3. Best global threshold by validation macro-F1
    Y_test_pred_best_macro = binarize_global(P_test, best_macro_threshold)

    metrics_best_macro, per_class_best_macro = evaluate_and_save(
        Y_true=Y_test,
        P_score=P_test,
        Y_pred=Y_test_pred_best_macro,
        label_map=label_map,
        threshold_name=f"best_val_macro_f1_threshold_{best_macro_threshold}",
        output_txt_path=output_dir / "test_metrics_best_macro_threshold.txt",
        output_csv_path=output_dir / "per_class_test_metrics_best_macro.csv",
    )

    # 4. Per-class thresholds selected on validation set
    Y_test_pred_per_class = binarize_per_class(P_test, best_thresholds)

    metrics_per_class, per_class_per_class = evaluate_and_save(
        Y_true=Y_test,
        P_score=P_test,
        Y_pred=Y_test_pred_per_class,
        label_map=label_map,
        threshold_name="per_class_thresholds_selected_on_validation",
        output_txt_path=output_dir / "test_metrics_per_class_thresholds.txt",
        output_csv_path=output_dir / "per_class_test_metrics_per_class.csv",
    )

    # Save binary predictions
    np.save(output_dir / "Y_test_pred_threshold_0_5.npy", Y_test_pred_05)
    np.save(output_dir / "Y_test_pred_best_micro_threshold.npy", Y_test_pred_best_micro)
    np.save(output_dir / "Y_test_pred_best_macro_threshold.npy", Y_test_pred_best_macro)
    np.save(output_dir / "Y_test_pred_per_class_thresholds.npy", Y_test_pred_per_class)

    # Save summary comparison
    comparison_df = pd.DataFrame([
        metrics_05,
        metrics_best_micro,
        metrics_best_macro,
        metrics_per_class,
    ])

    comparison_df.to_csv(output_dir / "test_threshold_comparison.csv", index=False)

    print_metric_summary("Test metrics with threshold 0.5", metrics_05)
    print_metric_summary("Test metrics with best validation micro-F1 threshold", metrics_best_micro)
    print_metric_summary("Test metrics with best validation macro-F1 threshold", metrics_best_macro)
    print_metric_summary("Test metrics with per-class validation thresholds", metrics_per_class)

    print("\n========== Saved files ==========")
    print(output_dir / "val_predictions.npy")
    print(output_dir / "test_predictions_best.npy")
    print(output_dir / "val_threshold_search.csv")
    print(output_dir / "best_thresholds_per_class.csv")
    print(output_dir / "test_threshold_comparison.csv")
    print(output_dir / "per_class_test_metrics_0_5.csv")
    print(output_dir / "per_class_test_metrics_best_micro.csv")
    print(output_dir / "per_class_test_metrics_best_macro.csv")
    print(output_dir / "per_class_test_metrics_per_class.csv")

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()