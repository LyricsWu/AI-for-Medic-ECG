#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
split_dataset.py

把 output_multilabel/ 中的多标签 ECG 数据集划分为 train / val / test。

默认输入：
    output_multilabel/X_filtered.npy
    output_multilabel/Y_filtered.npy
    output_multilabel/metadata_filtered.csv
    output_multilabel/label_map.csv

默认输出：
    output_split/
        X_train.npy
        Y_train.npy
        X_val.npy
        Y_val.npy
        X_test.npy
        Y_test.npy
        metadata_train.csv
        metadata_val.csv
        metadata_test.csv
        label_map.csv
        split_indices.npz
        split_summary.txt
        split_config.json

默认比例：
    train: 70%
    val:   15%
    test:  15%

运行：
    python split_dataset.py

说明：
- 这是多标签任务。
- 严格 multilabel stratified split 需要额外库 iterative-stratification。
- 本脚本默认使用 sklearn 的 train_test_split，并用每条样本的标签数量做近似分层。
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def check_ratios(train_ratio, val_ratio, test_ratio):
    total = train_ratio + val_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio 必须等于 1，当前为 {total}"
        )

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("train_ratio, val_ratio, test_ratio 都必须大于 0")


def get_label_count_strata(Y):
    """
    使用每条样本的 positive label 数量作为近似分层标签。

    例如：
        样本 A 有 1 个标签 -> strata = 1
        样本 B 有 3 个标签 -> strata = 3
    """
    label_counts = Y.sum(axis=1).astype(int)
    return label_counts


def save_split_summary(
    summary_path,
    X,
    Y,
    metadata_df,
    label_map_df,
    train_idx,
    val_idx,
    test_idx,
    train_ratio,
    val_ratio,
    test_ratio,
    random_state,
):
    """
    保存划分摘要。
    """
    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_idx]

    label_counts_train = Y_train.sum(axis=0).astype(int)
    label_counts_val = Y_val.sum(axis=0).astype(int)
    label_counts_test = Y_test.sum(axis=0).astype(int)

    per_sample_train = Y_train.sum(axis=1)
    per_sample_val = Y_val.sum(axis=1)
    per_sample_test = Y_test.sum(axis=1)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("========== ECG Multi-label Split Summary ==========\n\n")

        f.write("Input dataset\n")
        f.write(f"X shape: {tuple(X.shape)}\n")
        f.write(f"X dtype: {X.dtype}\n")
        f.write(f"Y shape: {tuple(Y.shape)}\n")
        f.write(f"Y dtype: {Y.dtype}\n")
        f.write(f"metadata shape: {tuple(metadata_df.shape)}\n")
        f.write(f"label_map shape: {tuple(label_map_df.shape)}\n\n")

        f.write("Split config\n")
        f.write(f"train_ratio: {train_ratio}\n")
        f.write(f"val_ratio: {val_ratio}\n")
        f.write(f"test_ratio: {test_ratio}\n")
        f.write(f"random_state: {random_state}\n\n")

        f.write("Split sizes\n")
        f.write(f"train samples: {len(train_idx)}\n")
        f.write(f"val samples: {len(val_idx)}\n")
        f.write(f"test samples: {len(test_idx)}\n\n")

        f.write("Per-sample label count distribution\n")
        f.write("Train:\n")
        f.write(str(pd.Series(per_sample_train).value_counts().sort_index()))
        f.write("\n\nVal:\n")
        f.write(str(pd.Series(per_sample_val).value_counts().sort_index()))
        f.write("\n\nTest:\n")
        f.write(str(pd.Series(per_sample_test).value_counts().sort_index()))
        f.write("\n\n")

        f.write("Per-label counts\n")
        f.write("label_index,dx_code,total,train,val,test\n")

        total_counts = Y.sum(axis=0).astype(int)

        for i in range(Y.shape[1]):
            if "dx_code" in label_map_df.columns:
                dx_code = label_map_df.loc[i, "dx_code"]
            else:
                dx_code = "Unknown"

            f.write(
                f"{i},{dx_code},{int(total_counts[i])},"
                f"{int(label_counts_train[i])},"
                f"{int(label_counts_val[i])},"
                f"{int(label_counts_test[i])}\n"
            )


def print_split_report(Y, label_map_df, train_idx, val_idx, test_idx):
    """
    在终端打印划分检查结果。
    """
    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_idx]

    print("\n========== 划分结果 ==========")
    print("Train samples:", len(train_idx))
    print("Val samples:", len(val_idx))
    print("Test samples:", len(test_idx))

    print("\n========== Y shape ==========")
    print("Y_train:", Y_train.shape)
    print("Y_val:", Y_val.shape)
    print("Y_test:", Y_test.shape)

    print("\n========== 每个样本标签数量分布 ==========")
    print("Train:")
    print(pd.Series(Y_train.sum(axis=1)).value_counts().sort_index())

    print("\nVal:")
    print(pd.Series(Y_val.sum(axis=1)).value_counts().sort_index())

    print("\nTest:")
    print(pd.Series(Y_test.sum(axis=1)).value_counts().sort_index())

    print("\n========== 每个标签在 train / val / test 中的数量 ==========")

    if "dx_code" in label_map_df.columns:
        dx_codes = label_map_df["dx_code"].values
    else:
        dx_codes = ["Unknown"] * Y.shape[1]

    report_df = pd.DataFrame({
        "label_index": np.arange(Y.shape[1]),
        "dx_code": dx_codes,
        "total": Y.sum(axis=0).astype(int),
        "train": Y_train.sum(axis=0).astype(int),
        "val": Y_val.sum(axis=0).astype(int),
        "test": Y_test.sum(axis=0).astype(int),
    })

    print(report_df)

    print("\n========== 最终检查 ==========")
    print("Train 每个样本至少一个标签:", bool((Y_train.sum(axis=1) > 0).all()))
    print("Val 每个样本至少一个标签:", bool((Y_val.sum(axis=1) > 0).all()))
    print("Test 每个样本至少一个标签:", bool((Y_test.sum(axis=1) > 0).all()))

    print("Train 每个标签至少一个样本:", bool((Y_train.sum(axis=0) > 0).all()))
    print("Val 每个标签至少一个样本:", bool((Y_val.sum(axis=0) > 0).all()))
    print("Test 每个标签至少一个样本:", bool((Y_test.sum(axis=0) > 0).all()))


def main():
    parser = argparse.ArgumentParser(
        description="Split ECG multi-label dataset into train / val / test."
    )

    parser.add_argument(
        "--input-dir",
        default="output_multilabel",
        help="输入文件夹，默认 output_multilabel"
    )

    parser.add_argument(
        "--output-dir",
        default="output_split",
        help="输出文件夹，默认 output_split"
    )

    parser.add_argument(
        "--x-file",
        default="X_filtered.npy",
        help="X 文件名，默认 X_filtered.npy"
    )

    parser.add_argument(
        "--y-file",
        default="Y_filtered.npy",
        help="Y 文件名，默认 Y_filtered.npy"
    )

    parser.add_argument(
        "--metadata-file",
        default="metadata_filtered.csv",
        help="metadata 文件名，默认 metadata_filtered.csv"
    )

    parser.add_argument(
        "--label-map-file",
        default="label_map.csv",
        help="label_map 文件名，默认 label_map.csv"
    )

    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help="训练集比例，默认 0.70"
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例，默认 0.15"
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="测试集比例，默认 0.15"
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="随机种子，默认 42"
    )

    parser.add_argument(
        "--no-save-x",
        action="store_true",
        help="不保存 X_train/X_val/X_test，只保存 Y、metadata 和 indices"
    )

    parser.add_argument(
        "--keep-float64",
        action="store_true",
        help="默认把 X 保存为 float32；加这个参数则保持原 dtype"
    )

    args = parser.parse_args()

    check_ratios(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    x_path = input_dir / args.x_file
    y_path = input_dir / args.y_file
    metadata_path = input_dir / args.metadata_file
    label_map_path = input_dir / args.label_map_file

    print("========== 输入文件 ==========")
    print("X:", x_path)
    print("Y:", y_path)
    print("metadata:", metadata_path)
    print("label_map:", label_map_path)

    if not x_path.exists():
        raise FileNotFoundError(f"找不到 X 文件: {x_path}")

    if not y_path.exists():
        raise FileNotFoundError(f"找不到 Y 文件: {y_path}")

    if not metadata_path.exists():
        raise FileNotFoundError(f"找不到 metadata 文件: {metadata_path}")

    if not label_map_path.exists():
        raise FileNotFoundError(f"找不到 label_map 文件: {label_map_path}")

    # =========================
    # 1. 读取数据
    # =========================
    print("\n========== 读取数据 ==========")

    X = np.load(x_path)
    Y = np.load(y_path)
    metadata_df = pd.read_csv(metadata_path)
    label_map_df = pd.read_csv(label_map_path)

    print("X shape:", X.shape)
    print("X dtype:", X.dtype)
    print("Y shape:", Y.shape)
    print("Y dtype:", Y.dtype)
    print("metadata shape:", metadata_df.shape)
    print("label_map shape:", label_map_df.shape)

    if len(X) != len(Y) or len(X) != len(metadata_df):
        raise ValueError(
            f"样本数不一致: len(X)={len(X)}, len(Y)={len(Y)}, "
            f"len(metadata)={len(metadata_df)}"
        )

    if Y.ndim != 2:
        raise ValueError(f"Y 必须是二维 multi-hot 矩阵，当前 Y.ndim={Y.ndim}")

    if Y.shape[1] != len(label_map_df):
        raise ValueError(
            f"Y 标签数和 label_map 行数不一致: "
            f"Y.shape[1]={Y.shape[1]}, len(label_map)={len(label_map_df)}"
        )

    if not (Y.sum(axis=1) > 0).all():
        raise ValueError("存在没有任何标签的样本，请先检查 Y_filtered.npy")

    # =========================
    # 2. 生成索引并划分
    # =========================
    print("\n========== 划分数据 ==========")

    indices = np.arange(len(X))
    strata = get_label_count_strata(Y)

    # 第一步：划分 train 和 temp，temp = val + test
    temp_ratio = args.val_ratio + args.test_ratio

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=args.random_state,
        shuffle=True,
        stratify=strata,
    )

    # 第二步：把 temp 划分成 val 和 test
    relative_test_ratio = args.test_ratio / temp_ratio
    temp_strata = strata[temp_idx]

    # 第二次划分 val/test 时，检查 temp_strata 中每个类别是否至少有 2 个样本
    # 如果某个类别只有 1 个样本，sklearn 的 stratify 会报错
    temp_strata_counts = pd.Series(temp_strata).value_counts()

    if temp_strata_counts.min() < 2:
        print("\n[提示] temp 集中某些标签数量分层类别少于 2 个样本。")
        print("[提示] 第二次 val/test 划分将不使用 stratify，改用普通随机划分。")
        print("temp_strata_counts:")
        print(temp_strata_counts.sort_index())

        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_ratio,
            random_state=args.random_state,
            shuffle=True,
            stratify=None,
        )
    else:
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=relative_test_ratio,
            random_state=args.random_state,
            shuffle=True,
            stratify=temp_strata,
        )

    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)
    test_idx = np.sort(test_idx)

    # =========================
    # 3. 创建划分后的数据
    # =========================
    print("\n========== 创建划分数据 ==========")

    Y_train = Y[train_idx]
    Y_val = Y[val_idx]
    Y_test = Y[test_idx]

    metadata_train = metadata_df.iloc[train_idx].reset_index(drop=True)
    metadata_val = metadata_df.iloc[val_idx].reset_index(drop=True)
    metadata_test = metadata_df.iloc[test_idx].reset_index(drop=True)

    if not args.no_save_x:
        X_train = X[train_idx]
        X_val = X[val_idx]
        X_test = X[test_idx]

        if not args.keep_float64:
            if X_train.dtype != np.float32:
                X_train = X_train.astype(np.float32)

            if X_val.dtype != np.float32:
                X_val = X_val.astype(np.float32)

            if X_test.dtype != np.float32:
                X_test = X_test.astype(np.float32)

        print("X_train shape:", X_train.shape, "dtype:", X_train.dtype)
        print("X_val shape:", X_val.shape, "dtype:", X_val.dtype)
        print("X_test shape:", X_test.shape, "dtype:", X_test.dtype)

    else:
        X_train = None
        X_val = None
        X_test = None

        print("使用 --no-save-x：不创建和保存 X_train/X_val/X_test")

    print("Y_train shape:", Y_train.shape)
    print("Y_val shape:", Y_val.shape)
    print("Y_test shape:", Y_test.shape)

    # =========================
    # 4. 保存文件
    # =========================
    print("\n========== 保存文件 ==========")

    if not args.no_save_x:
        np.save(output_dir / "X_train.npy", X_train)
        np.save(output_dir / "X_val.npy", X_val)
        np.save(output_dir / "X_test.npy", X_test)

        print("已保存 X_train/X_val/X_test")

    np.save(output_dir / "Y_train.npy", Y_train)
    np.save(output_dir / "Y_val.npy", Y_val)
    np.save(output_dir / "Y_test.npy", Y_test)

    metadata_train.to_csv(output_dir / "metadata_train.csv", index=False)
    metadata_val.to_csv(output_dir / "metadata_val.csv", index=False)
    metadata_test.to_csv(output_dir / "metadata_test.csv", index=False)

    label_map_df.to_csv(output_dir / "label_map.csv", index=False)

    np.savez(
        output_dir / "split_indices.npz",
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    config = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "x_file": args.x_file,
        "y_file": args.y_file,
        "metadata_file": args.metadata_file,
        "label_map_file": args.label_map_file,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": args.test_ratio,
        "random_state": args.random_state,
        "no_save_x": args.no_save_x,
        "keep_float64": args.keep_float64,
        "X_shape": tuple(X.shape),
        "Y_shape": tuple(Y.shape),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "test_samples": int(len(test_idx)),
    }

    with open(output_dir / "split_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    save_split_summary(
        summary_path=output_dir / "split_summary.txt",
        X=X,
        Y=Y,
        metadata_df=metadata_df,
        label_map_df=label_map_df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )

    print("已保存:", output_dir / "Y_train.npy")
    print("已保存:", output_dir / "Y_val.npy")
    print("已保存:", output_dir / "Y_test.npy")
    print("已保存:", output_dir / "metadata_train.csv")
    print("已保存:", output_dir / "metadata_val.csv")
    print("已保存:", output_dir / "metadata_test.csv")
    print("已保存:", output_dir / "label_map.csv")
    print("已保存:", output_dir / "split_indices.npz")
    print("已保存:", output_dir / "split_config.json")
    print("已保存:", output_dir / "split_summary.txt")

    # =========================
    # 5. 打印检查报告
    # =========================
    print_split_report(
        Y=Y,
        label_map_df=label_map_df,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
    )

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    main()