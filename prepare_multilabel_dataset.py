#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
prepare_multilabel_dataset.py

功能：
1. 读取 output/X.npy
2. 读取 output/metadata.csv
3. 读取 output/dx_counts.csv
4. 根据 Dx 出现次数筛选高频标签
5. 把 Dx 转换成 multi-hot 多标签矩阵 Y
6. 删除没有目标标签的样本
7. 保存过滤后的 X、Y、metadata 和 label_map

默认输入：
    output/X.npy
    output/metadata.csv
    output/dx_counts.csv

默认输出：
    output_multilabel/X_filtered.npy
    output_multilabel/Y_filtered.npy
    output_multilabel/metadata_filtered.csv
    output_multilabel/label_map.csv
    output_multilabel/dataset_summary.txt

默认筛选：
    只保留出现次数 >= 200 的 Dx 标签

运行：
    python prepare_multilabel_dataset.py

其他运行方式：
    python prepare_multilabel_dataset.py --min-count 500
    python prepare_multilabel_dataset.py --top-k 20
    python prepare_multilabel_dataset.py --y-dtype int8
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_dx_string(dx_value):
    """
    把 metadata.csv 里的 dx 字符串解析成列表。

    例如：
        "164889003,59118001,164934002"

    变成：
        ["164889003", "59118001", "164934002"]
    """
    if pd.isna(dx_value):
        return []

    dx_str = str(dx_value).strip()

    if dx_str == "":
        return []

    if dx_str.lower() in ["unknown", "nan", "none"]:
        return []

    dx_codes = []

    for code in dx_str.split(","):
        code = code.strip()
        if code:
            dx_codes.append(code)

    return dx_codes


def choose_labels(dx_counts_df, min_count=200, top_k=None):
    """
    从 dx_counts.csv 中选择目标标签。

    如果设置 top_k：
        选择出现次数最多的前 top_k 个标签

    如果没有设置 top_k：
        选择 count >= min_count 的标签
    """
    df = dx_counts_df.copy()

    df["dx_code"] = df["dx_code"].astype(str)
    df = df.sort_values("count", ascending=False).reset_index(drop=True)

    if top_k is not None:
        selected = df.head(top_k).copy()
    else:
        selected = df[df["count"] >= min_count].copy()

    selected = selected.reset_index(drop=True)
    selected["label_index"] = np.arange(len(selected))

    return selected


def build_multihot_labels(metadata_df, selected_labels_df):
    """
    构建多标签 multi-hot 矩阵。

    假设选中了 20 个标签，那么每条 ECG 的标签是长度为 20 的向量。

    例如某条 ECG 有第 0、3、5 个标签：
        [1, 0, 0, 1, 0, 1, 0, ...]

    返回：
        Y:
            shape = (样本数, 标签数)

        matched_label_counts:
            每条样本命中了几个目标标签

        all_dx_codes:
            每条样本原始解析出来的全部 Dx code
    """
    label_to_index = {}

    for _, row in selected_labels_df.iterrows():
        dx_code = str(row["dx_code"])
        label_index = int(row["label_index"])
        label_to_index[dx_code] = label_index

    n_samples = len(metadata_df)
    n_labels = len(selected_labels_df)

    Y = np.zeros((n_samples, n_labels), dtype=np.float32)

    matched_label_counts = np.zeros(n_samples, dtype=np.int32)
    all_dx_codes = []

    for i, dx_value in enumerate(metadata_df["dx"]):
        dx_codes = parse_dx_string(dx_value)
        all_dx_codes.append(dx_codes)

        for code in dx_codes:
            if code in label_to_index:
                j = label_to_index[code]
                Y[i, j] = 1.0

        matched_label_counts[i] = int(Y[i].sum())

    return Y, matched_label_counts, all_dx_codes


def save_summary(
    summary_path,
    input_dir,
    output_dir,
    original_x_shape,
    filtered_x_shape,
    y_shape,
    selected_labels_df,
    n_total,
    n_kept,
    n_removed,
    min_count,
    top_k,
    positive_label_count,
):
    """
    保存一个文本版摘要，方便之后查看数据集设置。
    """
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== Multi-label ECG Dataset Summary ==========\n\n")

        f.write("Input / Output\n")
        f.write(f"Input dir: {input_dir}\n")
        f.write(f"Output dir: {output_dir}\n\n")

        f.write("Label selection\n")
        f.write(f"min_count: {min_count}\n")
        f.write(f"top_k: {top_k}\n")
        f.write(f"number of selected labels: {len(selected_labels_df)}\n\n")

        f.write("Dataset shape\n")
        f.write(f"Original X shape: {original_x_shape}\n")
        f.write(f"Filtered X shape: {filtered_x_shape}\n")
        f.write(f"Y shape: {y_shape}\n\n")

        f.write("Samples\n")
        f.write(f"Total samples: {n_total}\n")
        f.write(f"Kept samples: {n_kept}\n")
        f.write(f"Removed samples without selected labels: {n_removed}\n")
        f.write(f"Total positive labels in Y: {positive_label_count}\n\n")

        f.write("Selected labels\n")
        for _, row in selected_labels_df.iterrows():
            f.write(
                f"label_index={int(row['label_index'])}, "
                f"dx_code={row['dx_code']}, "
                f"original_count={int(row['count'])}, "
                f"filtered_count={int(row['filtered_count'])}\n"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare multi-label ECG dataset."
    )

    parser.add_argument(
        "--input-dir",
        default="output",
        help="输入文件夹，默认 output"
    )

    parser.add_argument(
        "--output-dir",
        default="output_multilabel",
        help="输出文件夹，默认 output_multilabel"
    )

    parser.add_argument(
        "--x-file",
        default="X.npy",
        help="ECG 数据文件名，默认 X.npy"
    )

    parser.add_argument(
        "--metadata-file",
        default="metadata.csv",
        help="metadata 文件名，默认 metadata.csv"
    )

    parser.add_argument(
        "--dx-counts-file",
        default="dx_counts.csv",
        help="Dx 统计文件名，默认 dx_counts.csv"
    )

    parser.add_argument(
        "--min-count",
        type=int,
        default=200,
        help="选择出现次数 >= min_count 的 Dx 标签，默认 200"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="选择出现次数最多的前 K 个标签。如果设置，会覆盖 min-count"
    )

    parser.add_argument(
        "--y-dtype",
        choices=["float32", "int8"],
        default="float32",
        help="Y 标签矩阵保存格式，默认 float32，也可以选 int8"
    )

    parser.add_argument(
        "--keep-float64",
        action="store_true",
        help="默认会把 X 转成 float32；如果加这个参数，则保留原 dtype"
    )

    parser.add_argument(
        "--no-save-x",
        action="store_true",
        help="不保存 X_filtered.npy，只保存 Y 和 metadata"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / args.x_file
    metadata_path = input_dir / args.metadata_file
    dx_counts_path = input_dir / args.dx_counts_file

    print("========== 输入文件 ==========")
    print("X file:", x_path)
    print("metadata file:", metadata_path)
    print("dx_counts file:", dx_counts_path)

    if not x_path.exists():
        raise FileNotFoundError(f"找不到 X 文件: {x_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 metadata 文件: {metadata_path}")

    if not dx_counts_path.exists():
        raise FileNotFoundError(f"找不到 dx_counts 文件: {dx_counts_path}")

    # =========================
    # 1. 读取输入数据
    # =========================
    print("\n========== 读取数据 ==========")

    X = np.load(x_path)
    metadata_df = pd.read_csv(metadata_path)
    dx_counts_df = pd.read_csv(dx_counts_path)

    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("metadata shape:", metadata_df.shape)
    print("dx_counts shape:", dx_counts_df.shape)

    if len(X) != len(metadata_df):
        raise ValueError(
            f"X 和 metadata 数量不一致: len(X)={len(X)}, len(metadata)={len(metadata_df)}"
        )

    if "dx" not in metadata_df.columns:
        raise ValueError("metadata.csv 里面没有 dx 列")

    # =========================
    # 2. 选择目标标签
    # =========================
    print("\n========== 选择目标标签 ==========")

    selected_labels_df = choose_labels(
        dx_counts_df=dx_counts_df,
        min_count=args.min_count,
        top_k=args.top_k,
    )

    if len(selected_labels_df) == 0:
        raise ValueError("没有选中任何标签。请降低 --min-count 或使用 --top-k。")

    print("选中标签数量:", len(selected_labels_df))
    print(selected_labels_df[["label_index", "dx_code", "count"]])

    # =========================
    # 3. 构建 Y multi-hot 矩阵
    # =========================
    print("\n========== 构建 multi-hot 标签 ==========")

    Y, matched_label_counts, all_dx_codes = build_multihot_labels(
        metadata_df=metadata_df,
        selected_labels_df=selected_labels_df,
    )

    if args.y_dtype == "int8":
        Y = Y.astype(np.int8)
    else:
        Y = Y.astype(np.float32)

    metadata_df = metadata_df.copy()
    metadata_df["dx_list"] = ["|".join(codes) for codes in all_dx_codes]
    metadata_df["matched_label_count"] = matched_label_counts

    keep_mask = matched_label_counts > 0

    n_total = len(metadata_df)
    n_kept = int(keep_mask.sum())
    n_removed = int(n_total - n_kept)

    print("总样本数:", n_total)
    print("保留样本数:", n_kept)
    print("移除样本数，没有任何目标标签:", n_removed)
    print("原始 Y shape:", Y.shape)

    # =========================
    # 4. 过滤数据
    # =========================
    print("\n========== 过滤数据 ==========")

    X_filtered = X[keep_mask]
    Y_filtered = Y[keep_mask]
    metadata_filtered_df = metadata_df.loc[keep_mask].reset_index(drop=True)

    if not args.keep_float64:
        if X_filtered.dtype != np.float32:
            X_filtered = X_filtered.astype(np.float32)

    print("过滤后 X shape:", X_filtered.shape)
    print("过滤后 X dtype:", X_filtered.dtype)
    print("过滤后 Y shape:", Y_filtered.shape)
    print("过滤后 Y dtype:", Y_filtered.dtype)
    print("过滤后 metadata shape:", metadata_filtered_df.shape)

    # 过滤后的每类标签数量
    filtered_counts = Y_filtered.sum(axis=0).astype(int)

    selected_labels_df = selected_labels_df.copy()
    selected_labels_df["filtered_count"] = filtered_counts

    # =========================
    # 5. 保存文件
    # =========================
    print("\n========== 保存文件 ==========")

    x_out_path = output_dir / "X_filtered.npy"
    y_out_path = output_dir / "Y_filtered.npy"
    metadata_out_path = output_dir / "metadata_filtered.csv"
    label_map_path = output_dir / "label_map.csv"
    selected_dx_counts_path = output_dir / "selected_dx_counts.csv"
    summary_path = output_dir / "dataset_summary.txt"
    config_path = output_dir / "prepare_config.json"

    if not args.no_save_x:
        np.save(x_out_path, X_filtered)
        print("已保存:", x_out_path)
    else:
        print("跳过保存 X_filtered.npy")

    np.save(y_out_path, Y_filtered)
    metadata_filtered_df.to_csv(metadata_out_path, index=False)
    selected_labels_df.to_csv(label_map_path, index=False)
    selected_labels_df.to_csv(selected_dx_counts_path, index=False)

    print("已保存:", y_out_path)
    print("已保存:", metadata_out_path)
    print("已保存:", label_map_path)
    print("已保存:", selected_dx_counts_path)

    positive_label_count = int(Y_filtered.sum())

    save_summary(
        summary_path=summary_path,
        input_dir=input_dir,
        output_dir=output_dir,
        original_x_shape=tuple(X.shape),
        filtered_x_shape=tuple(X_filtered.shape),
        y_shape=tuple(Y_filtered.shape),
        selected_labels_df=selected_labels_df,
        n_total=n_total,
        n_kept=n_kept,
        n_removed=n_removed,
        min_count=args.min_count,
        top_k=args.top_k,
        positive_label_count=positive_label_count,
    )

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "x_file": args.x_file,
        "metadata_file": args.metadata_file,
        "dx_counts_file": args.dx_counts_file,
        "min_count": args.min_count,
        "top_k": args.top_k,
        "y_dtype": args.y_dtype,
        "keep_float64": args.keep_float64,
        "no_save_x": args.no_save_x,
        "original_x_shape": tuple(X.shape),
        "filtered_x_shape": tuple(X_filtered.shape),
        "y_filtered_shape": tuple(Y_filtered.shape),
        "n_total": n_total,
        "n_kept": n_kept,
        "n_removed": n_removed,
        "n_labels": int(len(selected_labels_df)),
        "positive_label_count": positive_label_count,
    }

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("已保存:", summary_path)
    print("已保存:", config_path)

    # =========================
    # 6. 最终检查
    # =========================
    print("\n========== 最终检查 ==========")

    every_sample_has_label = bool((Y_filtered.sum(axis=1) > 0).all())

    print("每个样本至少一个目标标签:", every_sample_has_label)
    print("Y 中 positive 标签总数:", positive_label_count)

    print("\n每类标签数量:")
    print(selected_labels_df[["label_index", "dx_code", "count", "filtered_count"]])

    print("\n每个样本命中的目标标签数量分布:")
    print(pd.Series(Y_filtered.sum(axis=1)).value_counts().sort_index())

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    main()