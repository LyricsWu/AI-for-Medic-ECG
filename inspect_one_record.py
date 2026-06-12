import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    # =========================
    # 1. 设置数据路径
    # =========================
    data_dir = Path("data")

    # 自动找 data 文件夹里的第一个 .hea 文件
    hea_files = sorted(data_dir.glob("*.hea"))

    if len(hea_files) == 0:
        print("没有在 data 文件夹中找到 .hea 文件")
        return

    first_hea = hea_files[0]

    # 去掉 .hea 后缀，得到 WFDB 读取需要的 record path
    # 例如 data/JS00001.hea -> data/JS00001
    record_path = first_hea.with_suffix("")

    print("========== 文件信息 ==========")
    print("First HEA file:", first_hea)
    print("Record path:", record_path)

    # 检查对应的 .mat 文件是否存在
    mat_path = record_path.with_suffix(".mat")

    if not mat_path.exists():
        print("找不到对应的 .mat 文件:", mat_path)
        return

    print("MAT file:", mat_path)

    # =========================
    # 2. 读取 WFDB 数据
    # =========================
    record = wfdb.rdrecord(str(record_path))

    signal = record.p_signal

    print("\n========== 基本信息 ==========")
    print("Record name:", record.record_name)
    print("Sampling frequency fs:", record.fs)
    print("Number of signals:", record.n_sig)
    print("Signal length:", record.sig_len)
    print("Signal shape:", signal.shape)
    print("Lead names:", record.sig_name)
    print("Units:", record.units)

    print("\n========== Comments ==========")
    if record.comments:
        for comment in record.comments:
            print(comment)
    else:
        print("No comments")

    # =========================
    # 3. 查看 .hea 原始内容
    # =========================
    print("\n========== HEA 原始内容 ==========")
    with open(first_hea, "r", encoding="utf-8", errors="ignore") as f:
        hea_content = f.read()
        print(hea_content)

    # =========================
    # 4. 转成 DataFrame 看前几行
    # =========================
    df = pd.DataFrame(signal, columns=record.sig_name)

    print("\n========== 前 10 行数据 ==========")
    print(df.head(10))

    print("\n========== 数据统计 ==========")
    print(df.describe())

    # =========================
    # 5. 简单质量检查
    # =========================
    print("\n========== 数据质量检查 ==========")
    print("NaN count:", np.isnan(signal).sum())
    print("Inf count:", np.isinf(signal).sum())
    print("Global min:", np.nanmin(signal))
    print("Global max:", np.nanmax(signal))
    print("Global mean:", np.nanmean(signal))
    print("Global std:", np.nanstd(signal))

    # =========================
    # 6. 画第 0 个通道前 10 秒并保存
    # =========================
    fs = record.fs
    seconds = 10
    channel = 0

    n = int(fs * seconds)
    n = min(n, signal.shape[0])

    t = np.arange(n) / fs

    plt.figure(figsize=(14, 4))
    plt.plot(t, signal[:n, channel])
    plt.xlabel("Time / s")
    plt.ylabel(record.units[channel])
    plt.title(f"{record.record_name} - {record.sig_name[channel]} - first {seconds}s")
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("JS00001_lead_I.png", dpi=300)

    plt.show()

    # =========================
    # 7. 画所有通道前 10 秒
    # =========================
    fig, axes = plt.subplots(record.n_sig, 1, figsize=(14, 2 * record.n_sig), sharex=True)

    # 如果只有一个通道，axes 不是 list，这里统一处理
    if record.n_sig == 1:
        axes = [axes]

    for i in range(record.n_sig):
        axes[i].plot(t, signal[:n, i])
        axes[i].set_ylabel(record.sig_name[i])
        axes[i].grid(True)

    axes[-1].set_xlabel("Time / s")
    fig.suptitle(f"{record.record_name} - all channels - first {seconds}s")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()