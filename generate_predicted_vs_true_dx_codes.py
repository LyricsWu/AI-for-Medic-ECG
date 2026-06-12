#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_predicted_vs_true_dx_codes.py

Generate sample-level comparison CSV and per-class evaluation plots for multi-label ECG predictions.

Inputs:
    prediction_output/predicted_dx_codes.csv
    output_split/Y_test.npy
    output_split/label_map.csv
    output_split/metadata_test.csv   optional but recommended

Outputs:
    prediction_output/predicted_vs_true_dx_codes.csv
    prediction_output/error_analysis_by_dx_code_from_predicted_vs_true.csv
    prediction_output/per_class_f1_bar_chart.png
    prediction_output/per_class_precision_recall_f1_bar_chart.png

Usage:
    python generate_predicted_vs_true_dx_codes.py
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def find_project_root():
    """
    Find project root robustly.

    This allows the script to work even if it is accidentally placed inside
    subfolders such as model_output/.
    """
    current = Path(__file__).resolve().parent

    for p in [current] + list(current.parents):
        if (p / "output_split").exists():
            return p

    # fallback: current working directory
    return Path.cwd()


def split_codes(value):
    """
    Split a pipe-separated code string into a list.
    Empty / NaN values return an empty list.
    """
    if pd.isna(value):
        return []

    value = str(value).strip()

    if value == "" or value.lower() in ["nan", "none", "unknown"]:
        return []

    return [x.strip() for x in value.split("|") if x.strip()]


def codes_from_multihot(row, label_map):
    """
    Convert a multi-hot row into Dx code list.
    """
    positive_indices = np.where(row == 1)[0]
    codes = label_map.loc[positive_indices, "dx_code"].astype(str).tolist()
    return codes


def compute_per_class_error_analysis(comparison_df):
    """
    Compute per-class TP / FP / FN / precision / recall / F1 from sample-level CSV.
    """
    tp_counter = Counter()
    fp_counter = Counter()
    fn_counter = Counter()
    true_counter = Counter()
    pred_counter = Counter()

    for _, row in comparison_df.iterrows():
        for code in split_codes(row["true_positive_codes"]):
            tp_counter[code] += 1
        for code in split_codes(row["false_positive_codes"]):
            fp_counter[code] += 1
        for code in split_codes(row["false_negative_codes"]):
            fn_counter[code] += 1
        for code in split_codes(row["true_dx_codes"]):
            true_counter[code] += 1
        for code in split_codes(row["predicted_dx_codes"]):
            pred_counter[code] += 1

    all_codes = sorted(
        set(true_counter)
        | set(pred_counter)
        | set(tp_counter)
        | set(fp_counter)
        | set(fn_counter)
    )

    rows = []

    for code in all_codes:
        tp = tp_counter[code]
        fp = fp_counter[code]
        fn = fn_counter[code]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        rows.append({
            "dx_code": code,
            "true_count": int(true_counter[code]),
            "pred_count": int(pred_counter[code]),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })

    per_class_df = pd.DataFrame(rows)
    per_class_df = per_class_df.sort_values("f1", ascending=False).reset_index(drop=True)

    return per_class_df


def plot_per_class_f1(per_class_df, output_path):
    """
    Generate horizontal per-class F1 bar chart.
    """
    plot_df = per_class_df.sort_values("f1", ascending=True).reset_index(drop=True)

    colors = []
    for f1 in plot_df["f1"]:
        if f1 < 0.4:
            colors.append("#d73027")
        elif f1 < 0.6:
            colors.append("#fc8d59")
        elif f1 < 0.8:
            colors.append("#91bfdb")
        else:
            colors.append("#1a9850")

    plt.figure(figsize=(11, 8))
    plt.barh(plot_df["dx_code"].astype(str), plot_df["f1"], color=colors)
    plt.xlabel("F1 score")
    plt.ylabel("Dx code")
    plt.title("Per-class F1 Scores with Per-class Thresholds")
    plt.xlim(0, 1.0)
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    for i, row in plot_df.iterrows():
        text_x = min(float(row["f1"]) + 0.015, 0.98)
        plt.text(
            text_x,
            i,
            f"{row['f1']:.3f}  n={int(row['true_count'])}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_f1(per_class_df, output_path):
    """
    Generate grouped bar chart for per-class precision, recall, and F1.
    """
    plot_df = per_class_df.sort_values("f1", ascending=False).reset_index(drop=True)

    x = np.arange(len(plot_df))
    width = 0.26

    plt.figure(figsize=(15, 7))
    plt.bar(x - width, plot_df["precision"], width, label="Precision")
    plt.bar(x, plot_df["recall"], width, label="Recall")
    plt.bar(x + width, plot_df["f1"], width, label="F1")

    plt.xticks(x, plot_df["dx_code"].astype(str), rotation=60, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Score")
    plt.xlabel("Dx code")
    plt.title("Per-class Precision, Recall, and F1")
    plt.grid(axis="y", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    # =========================
    # 1. Set paths
    # =========================
    project_root = find_project_root()

    predicted_csv_path = project_root / "prediction_output" / "predicted_dx_codes.csv"
    y_test_path = project_root / "output_split" / "Y_test.npy"
    label_map_path = project_root / "output_split" / "label_map.csv"
    metadata_test_path = project_root / "output_split" / "metadata_test.csv"

    output_dir = project_root / "prediction_output"
    output_dir.mkdir(exist_ok=True)

    comparison_csv_path = output_dir / "predicted_vs_true_dx_codes.csv"
    per_class_csv_path = output_dir / "error_analysis_by_dx_code_from_predicted_vs_true.csv"
    f1_plot_path = output_dir / "per_class_f1_bar_chart.png"
    prf_plot_path = output_dir / "per_class_precision_recall_f1_bar_chart.png"

    # =========================
    # 2. Load files
    # =========================
    print("========== Loading files ==========")
    print("project_root:", project_root)
    print("predicted_csv:", predicted_csv_path)
    print("Y_test:", y_test_path)
    print("label_map:", label_map_path)
    print("metadata_test:", metadata_test_path)

    if not predicted_csv_path.exists():
        raise FileNotFoundError(f"Cannot find predicted CSV: {predicted_csv_path}")

    if not y_test_path.exists():
        raise FileNotFoundError(f"Cannot find Y_test: {y_test_path}")

    if not label_map_path.exists():
        raise FileNotFoundError(f"Cannot find label_map: {label_map_path}")

    pred_df = pd.read_csv(predicted_csv_path)
    Y_test = np.load(y_test_path).astype(int)
    label_map = pd.read_csv(label_map_path)

    if metadata_test_path.exists():
        metadata_test = pd.read_csv(metadata_test_path)
    else:
        metadata_test = None
        print("[Warning] metadata_test.csv not found. record_name will be empty.")

    print("pred_df shape:", pred_df.shape)
    print("Y_test shape:", Y_test.shape)
    print("label_map shape:", label_map.shape)
    if metadata_test is not None:
        print("metadata_test shape:", metadata_test.shape)

    # =========================
    # 3. Basic checks
    # =========================
    if len(pred_df) != len(Y_test):
        raise ValueError(
            f"Length mismatch: predicted CSV has {len(pred_df)} rows, "
            f"but Y_test has {len(Y_test)} rows."
        )

    if Y_test.shape[1] != len(label_map):
        raise ValueError(
            f"Label mismatch: Y_test has {Y_test.shape[1]} labels, "
            f"but label_map has {len(label_map)} rows."
        )

    pred_df = pred_df.copy()
    pred_df["sample_index"] = pred_df["sample_index"].astype(int)
    pred_df = pred_df.sort_values("sample_index").reset_index(drop=True)

    expected_indices = np.arange(len(pred_df))
    if not np.array_equal(pred_df["sample_index"].values, expected_indices):
        print("[Warning] sample_index is not exactly 0..N-1. The script will still use row order.")

    # =========================
    # 4. Build comparison rows
    # =========================
    print("\n========== Building comparison CSV ==========")

    rows = []

    for i in range(len(pred_df)):
        sample_index = int(pred_df.loc[i, "sample_index"])

        true_codes = codes_from_multihot(Y_test[i], label_map)
        pred_codes = split_codes(pred_df.loc[i, "predicted_dx_codes"])

        true_set = set(true_codes)
        pred_set = set(pred_codes)

        tp_codes = sorted(true_set & pred_set)
        fp_codes = sorted(pred_set - true_set)
        fn_codes = sorted(true_set - pred_set)

        if metadata_test is not None and "record_name" in metadata_test.columns:
            record_name = str(metadata_test.loc[i, "record_name"])
        else:
            record_name = ""

        if metadata_test is not None and "dx" in metadata_test.columns:
            original_dx = str(metadata_test.loc[i, "dx"])
        else:
            original_dx = ""

        exact_match = int(true_set == pred_set)

        rows.append({
            "sample_index": sample_index,
            "record_name": record_name,
            "original_dx_from_metadata": original_dx,
            "num_true_labels": len(true_codes),
            "true_dx_codes": "|".join(true_codes),
            "num_predicted_labels": int(pred_df.loc[i, "num_predicted_labels"]),
            "predicted_dx_codes": pred_df.loc[i, "predicted_dx_codes"] if not pd.isna(pred_df.loc[i, "predicted_dx_codes"]) else "",
            "predicted_probabilities": pred_df.loc[i, "predicted_probabilities"] if not pd.isna(pred_df.loc[i, "predicted_probabilities"]) else "",
            "used_thresholds": pred_df.loc[i, "used_thresholds"] if not pd.isna(pred_df.loc[i, "used_thresholds"]) else "",
            "true_positive_codes": "|".join(tp_codes),
            "false_positive_codes": "|".join(fp_codes),
            "false_negative_codes": "|".join(fn_codes),
            "num_tp": len(tp_codes),
            "num_fp": len(fp_codes),
            "num_fn": len(fn_codes),
            "exact_match": exact_match,
        })

    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(comparison_csv_path, index=False)

    print("Saved:", comparison_csv_path)
    print("comparison_df shape:", comparison_df.shape)

    # =========================
    # 5. Per-class error analysis
    # =========================
    print("\n========== Computing per-class error analysis ==========")

    per_class_df = compute_per_class_error_analysis(comparison_df)
    per_class_df.to_csv(per_class_csv_path, index=False)

    print("Saved:", per_class_csv_path)
    print("per_class_df shape:", per_class_df.shape)

    # =========================
    # 6. Generate plots
    # =========================
    print("\n========== Generating plots ==========")

    plot_per_class_f1(per_class_df, f1_plot_path)
    plot_precision_recall_f1(per_class_df, prf_plot_path)

    print("Saved:", f1_plot_path)
    print("Saved:", prf_plot_path)

    # =========================
    # 7. Quick summary
    # =========================
    print("\n========== Summary ==========")
    print("Total samples:", len(comparison_df))
    print("Exact match samples:", int(comparison_df["exact_match"].sum()))
    print("Exact match ratio:", float(comparison_df["exact_match"].mean()))
    print("Total TP:", int(comparison_df["num_tp"].sum()))
    print("Total FP:", int(comparison_df["num_fp"].sum()))
    print("Total FN:", int(comparison_df["num_fn"].sum()))

    print("\nPredicted label count distribution:")
    print(comparison_df["num_predicted_labels"].value_counts().sort_index())

    print("\nTrue label count distribution:")
    print(comparison_df["num_true_labels"].value_counts().sort_index())

    print("\nTop 10 classes by F1:")
    print(per_class_df.head(10)[[
        "dx_code", "true_count", "precision", "recall", "f1"
    ]])

    print("\nBottom 10 classes by F1:")
    print(per_class_df.sort_values("f1", ascending=True).head(10)[[
        "dx_code", "true_count", "precision", "recall", "f1"
    ]])

    print("\n========== Done ==========")


if __name__ == "__main__":
    main()
