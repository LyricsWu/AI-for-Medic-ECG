import numpy as np
import pandas as pd

X = np.load("output_multilabel/X_filtered.npy")
Y = np.load("output_multilabel/Y_filtered.npy")
metadata = pd.read_csv("output_multilabel/metadata_filtered.csv")
label_map = pd.read_csv("output_multilabel/label_map.csv")

print("========== 基本信息 ==========")
print("X shape:", X.shape)
print("X dtype:", X.dtype)
print("Y shape:", Y.shape)
print("Y dtype:", Y.dtype)
print("metadata shape:", metadata.shape)
print("label_map shape:", label_map.shape)

print("\n========== X 数值范围 ==========")
print("X min:", X.min())
print("X max:", X.max())
print("X mean:", X.mean())
print("X std:", X.std())

print("\n========== Y 检查 ==========")
print("Y unique values:", np.unique(Y))
print("每个样本至少一个标签:", bool((Y.sum(axis=1) > 0).all()))
print("每个标签至少一个样本:", bool((Y.sum(axis=0) > 0).all()))

print("\n========== 每个样本标签数分布 ==========")
print(pd.Series(Y.sum(axis=1)).value_counts().sort_index())

print("\n========== 每个标签样本数 ==========")
label_counts = Y.sum(axis=0).astype(int)

result = pd.DataFrame({
    "label_index": label_map["label_index"],
    "dx_code": label_map["dx_code"],
    "count_from_label_map": label_map["filtered_count"],
    "count_from_Y": label_counts,
})

print(result)

print("\n========== 前 5 条 metadata ==========")
print(metadata.head())

