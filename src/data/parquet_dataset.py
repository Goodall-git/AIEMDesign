# dataset_em_parquet.py
# -*- coding: utf-8 -*-
"""
电磁拓扑 + 多 S 参数曲线的 Parquet 数据集。

数据约定（由 data/build_dataset.py 生成）
----------------------------------------
x  (topology) : 三个二值矩阵 TM2 / TM3 / V1 堆叠，形状 (3, 19, 15)
y  (s_mag_db) : S11 / S21 / S22 的幅度(dB)，统一频率轴，形状 (3, 101)

频率轴固定为 0.0 ~ 10.0 GHz 共 101 点（步长 0.1 GHz），写在 Parquet 的
schema metadata 里（键 `freq_ghz`）。

注意：y 经过 y_transform 之后会被展平成 (3 * N,) 的一维向量，顺序为
[S11(N), S21(N), S22(N)]，与代理模型的 out_dim = 3 * N 对齐。
"""

import os
import logging
from functools import partial
from typing import Dict, Optional, Callable, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torch.utils import data
import pyarrow.parquet as pq

from src.registry import DATA

_logger = logging.getLogger(__name__)

META_SHAPE_SUFFIX = "_shape"
META_DTYPE_SUFFIX = "_dtype"
META_FREQ_KEY = "freq_ghz"

# 三个矩阵 / 三条 S 曲线的数量，用于校验 metadata
N_MATRICES = 3
N_S_PARAMS = 3


def scale_to_minus1_1(x):
    return x * 2.0 - 1.0


def downsample_s_curve(x, factor):
    """沿频率轴（最后一维）下采样，兼容 (N,) 与 (3, N)。"""
    return x[..., ::factor]


def normalize_s_curve_with_stats(x, mean, max_value, min_value):
    return (x - mean) / (max_value - min_value)


def _freq_to_index(freq, start_freq, end_freq, num_points):
    return int(round((freq - start_freq) * (num_points - 1) / (end_freq - start_freq)))


def frequency_select_s_curve_by_freq(
    x,
    original_start_freq,
    original_end_freq,
    original_num_points,
    crop_start_freq,
    crop_end_freq,
):
    """按频率裁剪曲线，作用在最后一维，兼容 (N,) 与 (3, N)。"""
    left = _freq_to_index(
        crop_start_freq,
        original_start_freq,
        original_end_freq,
        original_num_points,
    )
    right = _freq_to_index(
        crop_end_freq,
        original_start_freq,
        original_end_freq,
        original_num_points,
    ) + 1
    return x[..., left:right]


def get_x_transforms(enable_scale_to_minus1_1):
    ops = []
    if enable_scale_to_minus1_1:
        ops.append(scale_to_minus1_1)
    return T.Compose(ops) if ops else None


def get_y_transforms(
    downsample_factor,
    enable_frequency_crop=False,
    original_start_freq=0.0,
    original_end_freq=10.0,
    original_num_points=101,
    crop_start_freq=None,
    crop_end_freq=None,
    enable_stat_normalize=False,
    normalize_mean=0.0,
    normalize_max=1.0,
    normalize_min=0.0,
):
    """
    构造 y 的变换链，输入/输出都是 (3, N) 的 dB 张量。

    注意：enable_stat_normalize 会破坏 dB 的物理含义（reward 依赖真实 dB
    数值来判断指标），除非同步调整 reward 配置，否则请保持关闭。
    """
    ops = []
    if enable_frequency_crop:
        ops.append(
            partial(
                frequency_select_s_curve_by_freq,
                original_start_freq=original_start_freq,
                original_end_freq=original_end_freq,
                original_num_points=original_num_points,
                crop_start_freq=crop_start_freq,
                crop_end_freq=crop_end_freq,
            )
        )
    if downsample_factor > 1:
        ops.append(partial(downsample_s_curve, factor=downsample_factor))
    if enable_stat_normalize:
        ops.append(
            partial(
                normalize_s_curve_with_stats,
                mean=normalize_mean,
                max_value=normalize_max,
                min_value=normalize_min,
            )
        )
    return T.Compose(ops) if ops else None


def _parse_shape_and_dtype(meta: dict, key: str) -> Tuple[Tuple[int, ...], np.dtype]:
    shape_expr = meta[f"{key}{META_SHAPE_SUFFIX}"]
    dtype_expr = meta[f"{key}{META_DTYPE_SUFFIX}"]
    shape = tuple(eval(shape_expr))
    dtype = np.dtype(eval(f"np.{dtype_expr}"))
    return shape, dtype


def parse_freq_axis(meta: dict) -> Optional[np.ndarray]:
    """从 metadata 里取回统一频率轴（GHz）。"""
    raw = meta.get(META_FREQ_KEY)
    if raw is None:
        return None
    return np.asarray([float(v) for v in raw.split(",")], dtype=np.float64)


def read_parquet_metadata(parquet_path: str) -> Dict[str, str]:
    """读取 Parquet 的 schema metadata 并解码成 str -> str。"""
    schema = pq.read_schema(parquet_path)
    raw_meta = schema.metadata or {}
    return {k.decode("utf-8"): v.decode("utf-8") for k, v in raw_meta.items()}


def _to_array_by_metadata(x, shape: Tuple[int, ...], dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(x)
    if arr.dtype == object:
        arr = np.asarray(arr.tolist(), dtype=dtype)
    else:
        arr = arr.astype(dtype, copy=False)
    return arr.reshape(shape).copy()


def _build_reconstruct_transform(
    shape: Tuple[int, ...],
    np_dtype: np.dtype,
):
    def _fn(x):
        arr = _to_array_by_metadata(x, shape=shape, dtype=np_dtype)
        return torch.from_numpy(arr)
    return _fn


class EMTopologySCurveParquetDataset(Dataset):
    """
    PyTorch 风格的数据集类，用于读取电磁拓扑 + 多 S 参数曲线的 Parquet 数据集。

    Parquet 文件中每一行包含：
    - topology  : 3 x 19 x 15 的二值矩阵（TM2 / TM3 / V1），展平成 855 维
    - s_mag_db  : 3 x 101 的 dB 曲线（S11 / S21 / S22），展平成 303 维

    __getitem__ 返回：
    - topology_tensor : (3, 19, 15)，默认已缩放到 [-1, 1]
    - s_curve_tensor  : (3 * 101,) = (303,)，顺序 [S11, S21, S22]
    """

    def __init__(
        self,
        parquet_path: str,
        x_key: str,
        y_key: str,
        x_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        y_transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()

        self.parquet_path = parquet_path
        self.x_key = x_key
        self.y_key = y_key
        self.x_transform = x_transform
        self.y_transform = y_transform

        # 读取 Parquet 文件
        self.df = pd.read_parquet(self.parquet_path)
        self.meta = read_parquet_metadata(self.parquet_path)

        x_shape, x_np_dtype = _parse_shape_and_dtype(self.meta, self.x_key)
        y_shape, y_np_dtype = _parse_shape_and_dtype(self.meta, self.y_key)

        if len(x_shape) != 3 or x_shape[0] != N_MATRICES:
            raise ValueError(
                f"{self.x_key} 期望形状 (3, H, W)，metadata 给出 {x_shape}")
        if len(y_shape) != 2 or y_shape[0] != N_S_PARAMS:
            raise ValueError(
                f"{self.y_key} 期望形状 (3, N)，metadata 给出 {y_shape}")

        # 统一频率轴（GHz），供推理导出 / reward 校验使用
        self.freq_ghz = parse_freq_axis(self.meta)

        x_base_transform = _build_reconstruct_transform(
            shape=x_shape,
            np_dtype=x_np_dtype,
        )
        y_base_transform = _build_reconstruct_transform(
            shape=y_shape,
            np_dtype=y_np_dtype,
        )

        self.x_transform = T.Compose(
            [x_base_transform, self.x_transform]
            if self.x_transform is not None else [x_base_transform]
        )
        self.y_transform = T.Compose(
            [y_base_transform, self.y_transform]
            if self.y_transform is not None else [y_base_transform]
        )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        x_raw = row[self.x_key]
        y_raw = row[self.y_key]
        x = self.x_transform(x_raw)
        y = self.y_transform(y_raw)
        # (3, N) -> (3 * N,)，与代理模型的 out_dim 对齐
        y = y.reshape(-1)
        return x, y


def _em_parquet_dataloader_impl(
    root: str,
    x_key: str,
    y_key: str,
    batch_size: int,
    s_curve_downsample_factor: int,
    num_workers: int,
    x_enable_scale_to_minus1_1: bool = True,
    y_enable_frequency_crop: bool = False,
    y_original_start_freq: float = 0.0,
    y_original_end_freq: float = 10.0,
    y_original_num_points: int = 101,
    y_crop_start_freq: Optional[float] = None,
    y_crop_end_freq: Optional[float] = None,
    y_enable_stat_normalize: bool = False,
    y_normalize_mean: float = 0.0,
    y_normalize_max: float = 1.0,
    y_normalize_min: float = 0.0,
    train_filename: str = "train.parquet",
    val_filename: str = "val.parquet",
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = True,
    # 可选：与你现有训练框架对齐（如果你后续接 DDP sampler）
    train_sampler=None,
    val_sampler=None,
) -> Tuple[data.DataLoader, data.DataLoader]:
    """
    基于 EMTopologySCurveParquetDataset 构建 train/val 的 DataLoader。

    期望目录结构：
      root/
        train.parquet
        val.parquet

    返回：
      train_loader, val_loader
    """
    if persistent_workers is None:
        # num_workers>0 才能启用 persistent_workers
        persistent_workers = num_workers > 0

    train_path = os.path.join(root, train_filename)
    val_path = os.path.join(root, val_filename)

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"未找到训练集文件：{train_path}")
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"未找到验证集文件：{val_path}")

    x_transform = get_x_transforms(enable_scale_to_minus1_1=x_enable_scale_to_minus1_1)
    y_transform = get_y_transforms(
            downsample_factor=s_curve_downsample_factor,
            enable_frequency_crop=y_enable_frequency_crop,
            original_start_freq=y_original_start_freq,
            original_end_freq=y_original_end_freq,
            original_num_points=y_original_num_points,
            crop_start_freq=y_crop_start_freq,
            crop_end_freq=y_crop_end_freq,
            enable_stat_normalize=y_enable_stat_normalize,
            normalize_mean=y_normalize_mean,
            normalize_max=y_normalize_max,
            normalize_min=y_normalize_min,
        )

    trainset = EMTopologySCurveParquetDataset(
        parquet_path=train_path,
        x_key=x_key,
        y_key=y_key,
        x_transform=x_transform,
        y_transform=y_transform,
    )
    valset = EMTopologySCurveParquetDataset(
        parquet_path=val_path,
        x_key=x_key,
        y_key=y_key,
        x_transform=x_transform,
        y_transform=y_transform,
    )

    _logger.info(
        f"[EM] Loading Parquet datasets: "
        f"train(len={len(trainset)}) val(len={len(valset)})"
    )
    if trainset.freq_ghz is not None:
        _logger.info(
            f"[EM] frequency axis: {trainset.freq_ghz.size} points, "
            f"{trainset.freq_ghz[0]:.3f} ~ {trainset.freq_ghz[-1]:.3f} GHz"
        )

    train_loader = data.DataLoader(
        dataset=trainset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
    )

    val_loader = data.DataLoader(
        dataset=valset,
        batch_size=batch_size,
        shuffle=False,              # 验证集通常不 shuffle
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader


@DATA.register
def EMParquetDataloader(
    root: str,
    x_key: str,
    y_key: str,
    batch_size: int,
    s_curve_downsample_factor: int,
    num_workers: int,
    x_enable_scale_to_minus1_1: bool = True,
    y_enable_frequency_crop: bool = False,
    y_original_start_freq: float = 0.0,
    y_original_end_freq: float = 10.0,
    y_original_num_points: int = 101,
    y_crop_start_freq: Optional[float] = None,
    y_crop_end_freq: Optional[float] = None,
    y_enable_stat_normalize: bool = False,
    y_normalize_mean: float = 0.0,
    y_normalize_max: float = 1.0,
    y_normalize_min: float = 0.0,
    train_filename: str = "train.parquet",
    val_filename: str = "val.parquet",
    pin_memory: bool = True,
    persistent_workers: Optional[bool] = None,
    drop_last: bool = True,
    **kwargs):
    kwargs.pop("config", None)
    return _em_parquet_dataloader_impl(
        root=root,
        x_key=x_key,
        y_key=y_key,
        batch_size=batch_size,
        s_curve_downsample_factor=s_curve_downsample_factor,
        num_workers=num_workers,
        x_enable_scale_to_minus1_1=x_enable_scale_to_minus1_1,
        y_enable_frequency_crop=y_enable_frequency_crop,
        y_original_start_freq=y_original_start_freq,
        y_original_end_freq=y_original_end_freq,
        y_original_num_points=y_original_num_points,
        y_crop_start_freq=y_crop_start_freq,
        y_crop_end_freq=y_crop_end_freq,
        y_enable_stat_normalize=y_enable_stat_normalize,
        y_normalize_mean=y_normalize_mean,
        y_normalize_max=y_normalize_max,
        y_normalize_min=y_normalize_min,
        train_filename=train_filename,
        val_filename=val_filename,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        **kwargs,
    )
