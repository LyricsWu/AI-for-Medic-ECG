#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_1d_cnn.py

用于训练 12 导联 ECG 多标签分类模型。

输入：
    output_split/X_train.npy
    output_split/Y_train.npy
    output_split/X_val.npy
    output_split/Y_val.npy
    output_split/X_test.npy
    output_split/Y_test.npy
    output_split/label_map.csv

输出：
    model_output/ecg_1d_cnn_best.keras
    model_output/ecg_1d_cnn_final.keras
    model_output/training_history.csv
    model_output/test_predictions.npy
    model_output/test_metrics.txt

任务类型：
    multi-label classification

模型：
    1D CNN

最后一层：
    sigmoid

loss:
    binary_crossentropy
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks


def normalize_per_sample_per_lead(X):
    """
    对每条 ECG、每个导联单独标准化。

    X shape:
        (样本数, 时间点, 导联数)

    对 axis=1 求均值和标准差，也就是沿时间维度标准化。
    """
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, keepdims=True) + 1e-8
    return (X - mean) / std


def build_1d_cnn(input_shape, num_classes):
    """
    构建简单 1D CNN 模型。
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv1D(32, kernel_size=7, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(64, kernel_size=7, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(256, kernel_size=5, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="sigmoid")(x)

    model = models.Model(inputs=inputs, outputs=outputs)

    return model


def main():
    input_dir = Path("output_split")
    output_dir = Path("model_output")
    output_dir.mkdir(exist_ok=True)

    print("========== 读取数据 ==========")

    X_train = np.load(input_dir / "X_train.npy")
    Y_train = np.load(input_dir / "Y_train.npy")

    X_val = np.load(input_dir / "X_val.npy")
    Y_val = np.load(input_dir / "Y_val.npy")

    X_test = np.load(input_dir / "X_test.npy")
    Y_test = np.load(input_dir / "Y_test.npy")

    label_map = pd.read_csv(input_dir / "label_map.csv")

    print("X_train:", X_train.shape, X_train.dtype)
    print("Y_train:", Y_train.shape, Y_train.dtype)
    print("X_val:", X_val.shape, X_val.dtype)
    print("Y_val:", Y_val.shape, Y_val.dtype)
    print("X_test:", X_test.shape, X_test.dtype)
    print("Y_test:", Y_test.shape, Y_test.dtype)
    print("label_map:", label_map.shape)

    print("\n========== 标准化 ==========")

    X_train = normalize_per_sample_per_lead(X_train).astype(np.float32)
    X_val = normalize_per_sample_per_lead(X_val).astype(np.float32)
    X_test = normalize_per_sample_per_lead(X_test).astype(np.float32)

    print("After normalization:")
    print("X_train mean:", X_train.mean(), "std:", X_train.std())
    print("X_val mean:", X_val.mean(), "std:", X_val.std())
    print("X_test mean:", X_test.mean(), "std:", X_test.std())

    input_shape = X_train.shape[1:]
    num_classes = Y_train.shape[1]

    print("\n========== 构建模型 ==========")
    print("input_shape:", input_shape)
    print("num_classes:", num_classes)

    model = build_1d_cnn(
        input_shape=input_shape,
        num_classes=num_classes,
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            tf.keras.metrics.AUC(name="auc", multi_label=True),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    model.summary()

    print("\n========== 开始训练 ==========")

    cb = [
        callbacks.ModelCheckpoint(
            filepath=output_dir / "ecg_1d_cnn_best.keras",
            monitor="val_auc",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_auc",
            mode="max",
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_auc",
            mode="max",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train,
        Y_train,
        validation_data=(X_val, Y_val),
        epochs=50,
        batch_size=32,
        callbacks=cb,
        verbose=1,
    )

    print("\n========== 保存训练结果 ==========")

    model.save(output_dir / "ecg_1d_cnn_final.keras")

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    print("已保存模型和 history")

    print("\n========== 测试集评估 ==========")

    test_results = model.evaluate(X_test, Y_test, verbose=1)

    metric_names = model.metrics_names

    with open(output_dir / "test_metrics.txt", "w", encoding="utf-8") as f:
        for name, value in zip(metric_names, test_results):
            line = f"{name}: {value}\n"
            print(line.strip())
            f.write(line)

    print("\n========== 保存测试集预测 ==========")

    Y_pred = model.predict(X_test, batch_size=32)
    np.save(output_dir / "test_predictions.npy", Y_pred)

    print("Y_pred shape:", Y_pred.shape)
    print("已保存:", output_dir / "test_predictions.npy")

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    main()