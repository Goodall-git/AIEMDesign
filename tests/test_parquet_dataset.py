from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchvision.transforms as T
from pyhocon import ConfigFactory

from src.data.parquet_dataset import (
    META_DTYPE_SUFFIX,
    META_SHAPE_SUFFIX,
    EMTopologySCurveParquetDataset,
    _em_parquet_dataloader_impl,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SURROGATE_CONF = PROJECT_ROOT / "conf" / "surrogate.conf"
DATA_CONF = ConfigFactory.parse_file(str(SURROGATE_CONF)).get_config("data")

X_KEY = DATA_CONF.get_string("x_key")
Y_KEY = DATA_CONF.get_string("y_key")
Y_NUM_POINTS = DATA_CONF.get_int("y_original_num_points")
BATCH_SIZE = DATA_CONF.get_int("batch_size")
NUM_WORKERS = 0

# 三个 19x15 的二值矩阵（TM2 / TM3 / V1）
N_MATRICES = 3
TOPO_SHAPE = (N_MATRICES, 19, 15)
TOPO_FLAT_LEN = int(np.prod(TOPO_SHAPE))

# 三条 S 曲线（S11 / S21 / S22），数据集返回的是展平后的 (3 * N,)
N_S_PARAMS = 3
Y_FLAT_LEN = N_S_PARAMS * Y_NUM_POINTS

TRAIN_FILENAME = DATA_CONF.get_string("train_filename")
VAL_FILENAME = DATA_CONF.get_string("val_filename")

X_SCALE = DATA_CONF.get_bool("x_enable_scale_to_minus1_1")
Y_ENABLE_CROP = DATA_CONF.get_bool("y_enable_frequency_crop")
Y_ORIGINAL_START = DATA_CONF.get_float("y_original_start_freq")
Y_ORIGINAL_END = DATA_CONF.get_float("y_original_end_freq")
Y_CROP_START = DATA_CONF.get_float("y_crop_start_freq")
Y_CROP_END = DATA_CONF.get_float("y_crop_end_freq")
Y_DOWNSAMPLE = DATA_CONF.get_int("s_curve_downsample_factor")
Y_ENABLE_STAT_NORMALIZE = DATA_CONF.get_bool("y_enable_stat_normalize")
Y_NORMALIZE_MEAN = DATA_CONF.get_float("y_normalize_mean")
Y_NORMALIZE_MAX = DATA_CONF.get_float("y_normalize_max")
Y_NORMALIZE_MIN = DATA_CONF.get_float("y_normalize_min")


def _write_flat_parquet(path: Path, num_rows: int) -> None:
    topology = np.random.randint(0, 2, size=(num_rows, TOPO_FLAT_LEN), dtype=np.int8)
    # dB 曲线，范围与 build_dataset.py 的截断区间一致
    s_curve = (np.random.rand(num_rows, Y_FLAT_LEN).astype(np.float32) * 60.0) - 60.0

    table = pa.Table.from_arrays(
        [
            pa.array(topology.tolist(), type=pa.list_(pa.int8(), TOPO_FLAT_LEN)),
            pa.array(s_curve.tolist(), type=pa.list_(pa.float32(), Y_FLAT_LEN)),
        ],
        names=[X_KEY, Y_KEY],
    )
    freq = np.linspace(Y_ORIGINAL_START, Y_ORIGINAL_END, Y_NUM_POINTS)
    table = table.replace_schema_metadata(
        {
            f"{X_KEY}{META_SHAPE_SUFFIX}".encode("utf-8"): str(list(TOPO_SHAPE)).replace(" ", "").encode("utf-8"),
            f"{X_KEY}{META_DTYPE_SUFFIX}".encode("utf-8"): b"int8",
            f"{Y_KEY}{META_SHAPE_SUFFIX}".encode("utf-8"): f"[{N_S_PARAMS},{Y_NUM_POINTS}]".encode("utf-8"),
            f"{Y_KEY}{META_DTYPE_SUFFIX}".encode("utf-8"): b"float32",
            b"freq_ghz": ",".join(f"{v:.6f}" for v in freq).encode("utf-8"),
        }
    )
    pq.write_table(table, path)


def _expected_crop_len(num_points, original_start, original_end, crop_start, crop_end):
    left = int(round((crop_start - original_start) * (num_points - 1) / (original_end - original_start)))
    right = int(round((crop_end - original_start) * (num_points - 1) / (original_end - original_start))) + 1
    return right - left


def _expected_y_len():
    """数据集返回的 y 是 (3, N) 展平后的长度。"""
    if Y_ENABLE_CROP:
        per_curve = _expected_crop_len(
            num_points=Y_NUM_POINTS,
            original_start=Y_ORIGINAL_START,
            original_end=Y_ORIGINAL_END,
            crop_start=Y_CROP_START,
            crop_end=Y_CROP_END,
        )
    else:
        per_curve = Y_NUM_POINTS
    if Y_DOWNSAMPLE > 1:
        per_curve = len(range(0, per_curve, Y_DOWNSAMPLE))
    return N_S_PARAMS * per_curve


def test_dataset_read_train_parquet_basic_sample(tmp_path: Path):
    train_parquet = tmp_path / TRAIN_FILENAME
    _write_flat_parquet(train_parquet, num_rows=max(BATCH_SIZE, 8))

    dataset = EMTopologySCurveParquetDataset(
        str(train_parquet),
        x_key=X_KEY,
        y_key=Y_KEY,
    )
    assert len(dataset) > 0
    # metadata 里的统一频率轴应当被解析出来
    assert dataset.freq_ghz is not None
    assert dataset.freq_ghz.size == Y_NUM_POINTS

    topology, s_curve = dataset[0]
    assert isinstance(topology, torch.Tensor)
    assert isinstance(s_curve, torch.Tensor)
    assert topology.shape == TOPO_SHAPE
    assert s_curve.shape == (Y_FLAT_LEN,)
    assert topology.dtype == torch.int8
    assert s_curve.dtype == torch.float32


def test_dataset_custom_transform_can_add_channel_dim(tmp_path: Path):
    train_parquet = tmp_path / TRAIN_FILENAME
    _write_flat_parquet(train_parquet, num_rows=max(BATCH_SIZE, 8))

    dataset = EMTopologySCurveParquetDataset(
        str(train_parquet),
        x_key=X_KEY,
        y_key=Y_KEY,
        x_transform=T.Lambda(lambda x: x.unsqueeze(0)),
    )

    topology, s_curve = dataset[0]
    assert topology.shape == (1, *TOPO_SHAPE)
    assert s_curve.shape == (Y_FLAT_LEN,)


def test_dataloader_train_val_basic_batch_shape(tmp_path: Path):
    _write_flat_parquet(tmp_path / TRAIN_FILENAME, num_rows=max(BATCH_SIZE, 8))
    _write_flat_parquet(tmp_path / VAL_FILENAME, num_rows=max(BATCH_SIZE, 8))

    train_loader, val_loader = _em_parquet_dataloader_impl(
        root=str(tmp_path),
        x_key=X_KEY,
        y_key=Y_KEY,
        batch_size=BATCH_SIZE,
        s_curve_downsample_factor=Y_DOWNSAMPLE,
        num_workers=NUM_WORKERS,
        x_enable_scale_to_minus1_1=X_SCALE,
        y_enable_frequency_crop=Y_ENABLE_CROP,
        y_original_start_freq=Y_ORIGINAL_START,
        y_original_end_freq=Y_ORIGINAL_END,
        y_original_num_points=Y_NUM_POINTS,
        y_crop_start_freq=Y_CROP_START,
        y_crop_end_freq=Y_CROP_END,
        y_enable_stat_normalize=Y_ENABLE_STAT_NORMALIZE,
        y_normalize_mean=Y_NORMALIZE_MEAN,
        y_normalize_max=Y_NORMALIZE_MAX,
        y_normalize_min=Y_NORMALIZE_MIN,
        train_filename=TRAIN_FILENAME,
        val_filename=VAL_FILENAME,
    )

    train_topology, train_s_curve = next(iter(train_loader))
    assert train_topology.shape == (BATCH_SIZE, *TOPO_SHAPE)
    assert train_s_curve.shape == (BATCH_SIZE, _expected_y_len())
    assert train_topology.dtype == torch.float32
    assert train_s_curve.dtype == torch.float32

    val_topology, val_s_curve = next(iter(val_loader))
    assert val_topology.shape == (BATCH_SIZE, *TOPO_SHAPE)
    assert val_s_curve.shape == (BATCH_SIZE, _expected_y_len())


def test_dataloader_s_curve_downsample_shape(tmp_path: Path):
    _write_flat_parquet(tmp_path / TRAIN_FILENAME, num_rows=max(BATCH_SIZE, 8))
    _write_flat_parquet(tmp_path / VAL_FILENAME, num_rows=max(BATCH_SIZE, 8))

    factor = max(2, Y_DOWNSAMPLE)
    train_loader, val_loader = _em_parquet_dataloader_impl(
        root=str(tmp_path),
        x_key=X_KEY,
        y_key=Y_KEY,
        batch_size=BATCH_SIZE,
        s_curve_downsample_factor=factor,
        num_workers=NUM_WORKERS,
        x_enable_scale_to_minus1_1=X_SCALE,
        y_enable_frequency_crop=False,
        y_original_start_freq=Y_ORIGINAL_START,
        y_original_end_freq=Y_ORIGINAL_END,
        y_original_num_points=Y_NUM_POINTS,
        y_crop_start_freq=Y_CROP_START,
        y_crop_end_freq=Y_CROP_END,
        y_enable_stat_normalize=Y_ENABLE_STAT_NORMALIZE,
        y_normalize_mean=Y_NORMALIZE_MEAN,
        y_normalize_max=Y_NORMALIZE_MAX,
        y_normalize_min=Y_NORMALIZE_MIN,
        train_filename=TRAIN_FILENAME,
        val_filename=VAL_FILENAME,
    )

    train_topology, train_s_curve = next(iter(train_loader))
    val_topology, val_s_curve = next(iter(val_loader))

    expected = N_S_PARAMS * len(range(0, Y_NUM_POINTS, factor))
    assert train_topology.shape == (BATCH_SIZE, *TOPO_SHAPE)
    assert val_topology.shape == (BATCH_SIZE, *TOPO_SHAPE)
    assert train_s_curve.shape == (BATCH_SIZE, expected)
    assert val_s_curve.shape == (BATCH_SIZE, expected)


def test_dataloader_s_curve_frequency_crop_shape(tmp_path: Path):
    _write_flat_parquet(tmp_path / TRAIN_FILENAME, num_rows=max(BATCH_SIZE, 8))
    _write_flat_parquet(tmp_path / VAL_FILENAME, num_rows=max(BATCH_SIZE, 8))

    crop_start = Y_CROP_START
    crop_end = Y_CROP_END
    train_loader, val_loader = _em_parquet_dataloader_impl(
        root=str(tmp_path),
        x_key=X_KEY,
        y_key=Y_KEY,
        batch_size=BATCH_SIZE,
        s_curve_downsample_factor=1,
        num_workers=NUM_WORKERS,
        y_enable_frequency_crop=True,
        y_original_start_freq=Y_ORIGINAL_START,
        y_original_end_freq=Y_ORIGINAL_END,
        y_original_num_points=Y_NUM_POINTS,
        y_crop_start_freq=crop_start,
        y_crop_end_freq=crop_end,
        y_enable_stat_normalize=Y_ENABLE_STAT_NORMALIZE,
        y_normalize_mean=Y_NORMALIZE_MEAN,
        y_normalize_max=Y_NORMALIZE_MAX,
        y_normalize_min=Y_NORMALIZE_MIN,
        train_filename=TRAIN_FILENAME,
        val_filename=VAL_FILENAME,
    )

    expected = N_S_PARAMS * _expected_crop_len(
        num_points=Y_NUM_POINTS,
        original_start=Y_ORIGINAL_START,
        original_end=Y_ORIGINAL_END,
        crop_start=crop_start,
        crop_end=crop_end,
    )

    _, train_s_curve = next(iter(train_loader))
    _, val_s_curve = next(iter(val_loader))
    assert train_s_curve.shape == (BATCH_SIZE, expected)
    assert val_s_curve.shape == (BATCH_SIZE, expected)
