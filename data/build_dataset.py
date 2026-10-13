#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 data/raw_data 下的 (npz, csv) 样本对转换成训练用的 Parquet 数据集。

每个样本
--------
x  (topology)  : npz 中的 TM2 / TM3 / V1 三个二值矩阵，形状 (3, 19, 15)
y  (s_mag_db)  : csv 中 S11 / S21 / S22 的幅度(dB)，重采样后形状 (3, 101)

关键处理
--------
1. 频率轴统一：原始 csv 的频率点数在 69 ~ 582 之间、共 554 种网格，
   但**全部都是 0 ~ 10 GHz**。这里统一线性插值到 101 点（步长 0.1 GHz）。
2. dB 截断：csv 的 S*_mag_db 精确等于 20*log10(|S|)，S21 最低可到 -260 dB
   （数值噪声）。统一截断到 [-60, 0] dB，避免离群值主导 L1 损失。
3. 划分：默认按 9:1 切分 train / val，固定随机种子保证可复现。

用法
----
python data/build_dataset.py
python data/build_dataset.py --raw-root data/raw_data --out-dir data \
    --val-ratio 0.1 --seed 42
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------------------------------------------------------- 常量定义
MATRIX_ORDER: Tuple[str, ...] = ("TM2", "TM3", "V1")     # x 的 3 个通道顺序
S_PARAM_ORDER: Tuple[str, ...] = ("S11", "S21", "S22")   # y 的 3 条曲线顺序

MATRIX_SHAPE: Tuple[int, int] = (19, 15)                 # 单个矩阵的高、宽
X_KEY: str = "topology"
Y_KEY: str = "s_mag_db"

FREQ_START_GHZ: float = 0.0
FREQ_END_GHZ: float = 10.0
FREQ_NUM_POINTS: int = 101
DB_FLOOR: float = -60.0                                  # dB 下限截断
DB_CEIL: float = 0.0

_INDEX_RE = re.compile(r"data(\d+)$")


# ---------------------------------------------------------------- 工具函数
def sample_index(path: str) -> int:
    """从 data123.npz / data123.csv 里取出数字编号。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _INDEX_RE.search(stem)
    if m is None:
        raise ValueError(f"无法从文件名解析样本编号：{path}")
    return int(m.group(1))


def target_freq_axis() -> np.ndarray:
    """统一的目标频率轴（GHz）。"""
    return np.linspace(FREQ_START_GHZ, FREQ_END_GHZ, FREQ_NUM_POINTS)


def load_topology(npz_path: str) -> np.ndarray:
    """读取 npz，按 MATRIX_ORDER 堆成 (3, 19, 15) 的 int8 二值矩阵。"""
    with np.load(npz_path) as data:
        missing = [k for k in MATRIX_ORDER if k not in data.files]
        if missing:
            raise KeyError(f"{npz_path} 缺少矩阵 {missing}，实际包含 {sorted(data.files)}")
        mats = []
        for key in MATRIX_ORDER:
            m = np.asarray(data[key])
            if m.shape != MATRIX_SHAPE:
                raise ValueError(f"{npz_path} 的 {key} 形状为 {m.shape}，期望 {MATRIX_SHAPE}")
            mats.append(m)
    x = np.stack(mats, axis=0).astype(np.int8, copy=False)
    if not np.isin(x, (0, 1)).all():
        raise ValueError(f"{npz_path} 的矩阵包含 0/1 之外的值")
    return x


def load_s_params(csv_path: str, grid: np.ndarray) -> np.ndarray:
    """读取 csv 的三条 dB 曲线，线性插值到统一频率轴，返回 (3, N) float32。"""
    df = pd.read_csv(csv_path)

    freq = df["freq_ghz"].to_numpy(dtype=np.float64)
    if freq.size < 2:
        raise ValueError(f"{csv_path} 频率点不足：{freq.size}")

    # 保证 xp 单调递增（正常数据本来就是），必要时排序
    order = np.argsort(freq, kind="stable")
    freq = freq[order]

    curves = []
    for name in S_PARAM_ORDER:
        col = f"{name}_mag_db"
        if col not in df.columns:
            raise KeyError(f"{csv_path} 缺少列 {col}")
        mag_db = df[col].to_numpy(dtype=np.float64)[order]
        # 线性插值到统一网格；np.interp 对超出范围的点取端点值（这里范围一致）
        curves.append(np.interp(grid, freq, mag_db))

    y = np.stack(curves, axis=0)
    return np.clip(y, DB_FLOOR, DB_CEIL).astype(np.float32, copy=False)


def collect_samples(raw_root: str) -> List[Tuple[int, str, str]]:
    """扫描 npz / csv 目录，返回按编号排序且两边都存在的样本列表。"""
    npz_dir = os.path.join(raw_root, "npz")
    csv_dir = os.path.join(raw_root, "csv")
    for d in (npz_dir, csv_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"找不到目录：{d}")

    npz_map: Dict[int, str] = {
        sample_index(f): os.path.join(npz_dir, f)
        for f in os.listdir(npz_dir) if f.endswith(".npz")
    }
    csv_map: Dict[int, str] = {
        sample_index(f): os.path.join(csv_dir, f)
        for f in os.listdir(csv_dir) if f.endswith(".csv")
    }

    only_npz = sorted(set(npz_map) - set(csv_map))
    only_csv = sorted(set(csv_map) - set(npz_map))
    if only_npz or only_csv:
        print(f"[WARN] npz 独有 {len(only_npz)} 个，csv 独有 {len(only_csv)} 个，已跳过",
              file=sys.stderr)

    common = sorted(set(npz_map) & set(csv_map))
    if not common:
        raise RuntimeError(f"{raw_root} 下没有可用的 npz/csv 配对")
    return [(i, npz_map[i], csv_map[i]) for i in common]


def build_arrays(samples: List[Tuple[int, str, str]], grid: np.ndarray):
    """把全部样本读成三个等长列表：样本号 / x 展平 / y 展平。"""
    x_flat = MATRIX_SHAPE[0] * MATRIX_SHAPE[1] * len(MATRIX_ORDER)
    y_flat = grid.size * len(S_PARAM_ORDER)

    ids: List[int] = []
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    log_every = max(1, len(samples) // 20)

    for n, (idx, npz_path, csv_path) in enumerate(samples, start=1):
        x = load_topology(npz_path)
        y = load_s_params(csv_path, grid)
        ids.append(idx)
        xs.append(np.ascontiguousarray(x.reshape(-1)))
        ys.append(np.ascontiguousarray(y.reshape(-1)))
        if n % log_every == 0 or n == len(samples):
            print(f"  已读取 {n}/{len(samples)} 个样本")

    X = np.stack(xs).astype(np.int8, copy=False)
    Y = np.stack(ys).astype(np.float32, copy=False)
    assert X.shape[1] == x_flat, X.shape
    assert Y.shape[1] == y_flat, Y.shape
    return np.asarray(ids, dtype=np.int32), X, Y


def build_metadata(grid: np.ndarray) -> Dict[bytes, bytes]:
    """写入 Parquet 的 schema metadata，供数据集重建张量时使用。"""
    meta = {
        "x_key": X_KEY,
        "y_key": Y_KEY,
        f"{X_KEY}_shape": str(list((len(MATRIX_ORDER),) + MATRIX_SHAPE)).replace(" ", ""),
        f"{X_KEY}_dtype": "int8",
        f"{Y_KEY}_shape": f"[{len(S_PARAM_ORDER)},{grid.size}]",
        f"{Y_KEY}_dtype": "float32",
        "matrix_order": ",".join(MATRIX_ORDER),
        "s_param_order": ",".join(S_PARAM_ORDER),
        "freq_start_ghz": repr(float(grid[0])),
        "freq_end_ghz": repr(float(grid[-1])),
        "freq_num_points": str(int(grid.size)),
        "freq_ghz": ",".join(f"{v:.6f}" for v in grid),
        "db_floor": repr(float(DB_FLOOR)),
        "db_ceil": repr(float(DB_CEIL)),
    }
    return {k.encode("utf-8"): v.encode("utf-8") for k, v in meta.items()}


def write_parquet(path: str, ids: np.ndarray, X: np.ndarray, Y: np.ndarray,
                  grid: np.ndarray) -> None:
    """按固定长度 list 类型写出 Parquet，并附带 shape/dtype metadata。"""
    table = pa.Table.from_arrays(
        [
            pa.array(ids.tolist(), type=pa.int32()),
            pa.array(X.tolist(), type=pa.list_(pa.int8(), X.shape[1])),
            pa.array(Y.tolist(), type=pa.list_(pa.float32(), Y.shape[1])),
        ],
        names=["sample_id", X_KEY, Y_KEY],
    )
    table = table.replace_schema_metadata(build_metadata(grid))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=True)
    print(f"  写出 {path}：{table.num_rows} 行，"
          f"x={X.shape[1]} 维，y={Y.shape[1]} 维，"
          f"{os.path.getsize(path) / 1e6:.1f} MB")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把 raw_data 的 npz/csv 转成 train/val Parquet 数据集")
    parser.add_argument("--raw-root", default=os.path.join("data", "raw_data"),
                        help="raw_data 目录（其下应有 npz/ 与 csv/）")
    parser.add_argument("--out-dir", default="data",
                        help="train.parquet / val.parquet 的输出目录")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--seed", type=int, default=42, help="划分随机种子")
    parser.add_argument("--limit", type=int, default=None,
                        help="仅转换前 N 个样本（调试用）")
    args = parser.parse_args()

    grid = target_freq_axis()
    samples = collect_samples(args.raw_root)
    if args.limit is not None:
        samples = samples[: args.limit]
    print(f"共 {len(samples)} 个样本，目标频率轴 {grid.size} 点 "
          f"({grid[0]:.1f} ~ {grid[-1]:.1f} GHz，步长 {grid[1] - grid[0]:.2f} GHz)")

    ids, X, Y = build_arrays(samples, grid)

    # 固定种子打乱后切分
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(ids))
    n_val = max(1, int(round(len(ids) * args.val_ratio)))
    val_pos, train_pos = perm[:n_val], perm[n_val:]

    write_parquet(os.path.join(args.out_dir, "train.parquet"),
                  ids[train_pos], X[train_pos], Y[train_pos], grid)
    write_parquet(os.path.join(args.out_dir, "val.parquet"),
                  ids[val_pos], X[val_pos], Y[val_pos], grid)

    band = (grid >= 6.4) & (grid <= 7.1)
    print(f"训练集 {train_pos.size} 个，验证集 {val_pos.size} 个")
    print(f"y(dB) 全量范围：[{Y.min():.3f}, {Y.max():.3f}]；"
          f"6.4~7.1 GHz 内 {int(band.sum())} 个频点")


if __name__ == "__main__":
    main()
