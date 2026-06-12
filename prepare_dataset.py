import wfdb
import numpy as np
import pandas as pd
from pathlib import Path


def parse_comments(comments):
    """
    解析 WFDB header 里的 comments。
    例如：
        Age: 85
        Sex: Male
        Dx: 164889003,59118001,164934002
    """
    metadata = {}

    for comment in comments:
        if ":" in comment:
            key, value = comment.split(":", 1)
            metadata[key.strip()] = value.strip()

    return metadata


def read_one_record(record_path):
    """
    读取单个 WFDB record。
    record_path 不带 .hea 或 .mat 后缀，例如 data/JS00001
    """
    record = wfdb.rdrecord(str(record_path))
    signal = record.p_signal

    metadata = parse_comments(record.comments)

    info = {
        "record_name": record.record_name,
        "record_path": str(record_path),
        "fs": record.fs,
        "n_sig": record.n_sig,
        "sig_len": record.sig_len,
        "shape": str(signal.shape) if signal is not None else None,
        "lead_names": ",".join(record.sig_name) if record.sig_name else "",
        "units": ",".join(record.units) if record.units else "",
        "age": metadata.get("Age", "Unknown"),
        "sex": metadata.get("Sex", "Unknown"),
        "dx": metadata.get("Dx", "Unknown"),
        "rx": metadata.get("Rx", "Unknown"),
        "hx": metadata.get("Hx", "Unknown"),
        "sx": metadata.get("Sx", "Unknown"),
    }

    return signal, info


def main():
    # =========================
    # 1. 设置路径
    # =========================
    data_dir = Path("data")
    output_dir = Path("output")

    output_dir.mkdir(exist_ok=True)

    # =========================
    # 2. 扫描所有 .hea 文件
    # =========================
    hea_files = sorted(data_dir.glob("*.hea"))

    if len(hea_files) == 0:
        print("没有在 data 文件夹中找到 .hea 文件")
        return

    print("========== 扫描结果 ==========")
    print(f"找到 .hea 文件数量: {len(hea_files)}")

    # =========================
    # 3. 批量读取
    # =========================
    all_signals = []
    all_metadata = []
    failed_records = []

    expected_shape = None

    for idx, hea_file in enumerate(hea_files):
        record_path = hea_file.with_suffix("")
        mat_file = record_path.with_suffix(".mat")

        print(f"\n[{idx + 1}/{len(hea_files)}] 正在读取: {record_path.name}")

        if not mat_file.exists():
            print(f"  跳过：找不到对应的 .mat 文件: {mat_file}")
            failed_records.append({
                "record_name": record_path.name,
                "reason": "missing_mat_file"
            })
            continue

        try:
            signal, info = read_one_record(record_path)

            if signal is None:
                print("  跳过：signal 为空")
                failed_records.append({
                    "record_name": record_path.name,
                    "reason": "empty_signal"
                })
                continue

            # 检查 shape 是否一致
            if expected_shape is None:
                expected_shape = signal.shape
                print(f"  设置 expected_shape = {expected_shape}")
            else:
                if signal.shape != expected_shape:
                    print(f"  警告：shape 不一致，当前 {signal.shape}，期望 {expected_shape}")
                    failed_records.append({
                        "record_name": record_path.name,
                        "reason": f"shape_mismatch_{signal.shape}"
                    })
                    continue

            # 检查 NaN / Inf
            nan_count = np.isnan(signal).sum()
            inf_count = np.isinf(signal).sum()

            info["nan_count"] = int(nan_count)
            info["inf_count"] = int(inf_count)
            info["global_min"] = float(np.nanmin(signal))
            info["global_max"] = float(np.nanmax(signal))
            info["global_mean"] = float(np.nanmean(signal))
            info["global_std"] = float(np.nanstd(signal))

            if nan_count > 0 or inf_count > 0:
                print(f"  警告：存在 NaN 或 Inf，NaN={nan_count}, Inf={inf_count}")

            all_signals.append(signal)
            all_metadata.append(info)

            print("  读取成功")
            print(f"  shape: {signal.shape}")
            print(f"  fs: {info['fs']}")
            print(f"  dx: {info['dx']}")

        except Exception as e:
            print(f"  读取失败: {repr(e)}")
            failed_records.append({
                "record_name": record_path.name,
                "reason": repr(e)
            })

    # =========================
    # 4. 保存结果
    # =========================
    print("\n========== 保存结果 ==========")

    if len(all_signals) == 0:
        print("没有成功读取任何样本")
        return

    # 转成 numpy array
    # 形状通常是: 样本数 x 5000 x 12
    X = np.stack(all_signals, axis=0)

    metadata_df = pd.DataFrame(all_metadata)
    failed_df = pd.DataFrame(failed_records)

    X_path = output_dir / "X.npy"
    metadata_path = output_dir / "metadata.csv"
    failed_path = output_dir / "failed_records.csv"

    np.save(X_path, X)
    metadata_df.to_csv(metadata_path, index=False)

    if len(failed_records) > 0:
        failed_df.to_csv(failed_path, index=False)

    print(f"成功读取样本数: {len(all_signals)}")
    print(f"失败样本数: {len(failed_records)}")
    print(f"X shape: {X.shape}")
    print(f"已保存: {X_path}")
    print(f"已保存: {metadata_path}")

    if len(failed_records) > 0:
        print(f"已保存: {failed_path}")

    # =========================
    # 5. 统计 Dx 标签
    # =========================
    print("\n========== Dx 标签统计 ==========")

    dx_counter = {}

    for dx_string in metadata_df["dx"]:
        if pd.isna(dx_string) or dx_string == "Unknown":
            dx_codes = ["Unknown"]
        else:
            dx_codes = str(dx_string).split(",")

        for code in dx_codes:
            code = code.strip()
            dx_counter[code] = dx_counter.get(code, 0) + 1

    dx_count_df = pd.DataFrame(
        sorted(dx_counter.items(), key=lambda x: x[1], reverse=True),
        columns=["dx_code", "count"]
    )

    dx_count_path = output_dir / "dx_counts.csv"
    dx_count_df.to_csv(dx_count_path, index=False)

    print(dx_count_df.head(20))
    print(f"已保存: {dx_count_path}")

    print("\n========== 完成 ==========")


if __name__ == "__main__":
    main()