## 数据说明（GDS / npz / csv → Parquet）

### 目录结构

```
data/
├── build_dataset.py          # 转换脚本（npz + csv → train/val.parquet）
├── train.parquet             # 训练集（由脚本生成）
├── val.parquet               # 验证集（由脚本生成）
└── raw_data/
    ├── npz/                  # data1.npz ... data4180.npz
    ├── csv/                  # data1.csv ... data4180.csv
    └── gds/                  # data1.gds ... data4180.gds（版图，训练流程不使用）
```

三个子目录按**相同的编号**一一对应，编号从 1 到 4180，共 4180 个样本。

---

### 原始数据格式

**`npz/dataN.npz`** —— 三个二值矩阵，作为模型输入（x）：

| 键 | 形状 | 类型 | 含义 |
|---|---|---|---|
| `TM2` | (19, 15) | uint8 | 第 2 层金属 |
| `TM3` | (19, 15) | uint8 | 第 3 层金属 |
| `V1`  | (19, 15) | uint8 | 过孔层（201 个样本为全 0） |

**`csv/dataN.csv`** —— 12 列：

```
dataset, freq_hz, freq_ghz,
S11_real, S11_imag, S11_mag_db,
S21_real, S21_imag, S21_mag_db,
S22_real, S22_imag, S22_mag_db
```

其中 `S*_mag_db` 精确等于 `20 * log10(|S|)`。

⚠️ **原始 csv 的频率轴不统一**：所有文件都是 0 ~ 10 GHz，但采样点数在 **69 ~ 582** 之间、
共 **554 种不同的频率网格**。因此必须重采样到统一网格才能做定长回归。

---

### 转换后的 Parquet 格式

脚本把每个样本整理成一行，两列固定长度数组 + schema metadata：

| 列 | 展平长度 | metadata 形状 | 类型 | 含义 |
|---|---|---|---|---|
| `topology` | 855 | `[3,19,15]` | int8 | 通道顺序 `TM2, TM3, V1` |
| `s_mag_db` | 303 | `[3,101]` | float32 | 顺序 `S11, S21, S22`，单位 dB |

写入 metadata 的键（`src/data/parquet_dataset.py` 会读取）：

| 键 | 值 |
|---|---|
| `topology_shape` / `topology_dtype` | `[3,19,15]` / `int8` |
| `s_mag_db_shape` / `s_mag_db_dtype` | `[3,101]` / `float32` |
| `matrix_order` | `TM2,TM3,V1` |
| `s_param_order` | `S11,S21,S22` |
| `freq_start_ghz` / `freq_end_ghz` / `freq_num_points` | `0.0` / `10.0` / `101` |
| `freq_ghz` | 101 个频率值，逗号分隔 |
| `db_floor` / `db_ceil` | `-60.0` / `0.0` |

**处理规则**

1. **频率重采样**：统一线性插值到 0.0 ~ 10.0 GHz 共 101 点（步长 0.1 GHz）。
2. **dB 截断**：截断到 `[-60, 0]` dB。原始 `S21_mag_db` 最低可到 -260 dB（数值噪声），
   不截断会让 L1 损失被离群值主导。
3. **划分**：固定随机种子（默认 42）打乱后按 9:1 切分，得到 3762 / 418。

---

### 使用方法

```bash
# 依赖：numpy pandas pyarrow
python data/build_dataset.py

# 常用参数
python data/build_dataset.py \
  --raw-root data/raw_data \
  --out-dir data \
  --val-ratio 0.1 \
  --seed 42
```

脚本会打印读取进度，并在结束时输出训练/验证集样本数、y 的取值范围以及
6.4 ~ 7.1 GHz 频段内的频点数。

> 修改 `build_dataset.py` 里的 `FREQ_NUM_POINTS` / `FREQ_START_GHZ` / `FREQ_END_GHZ`
> 之后，必须同步修改 `conf/surrogate.conf` 的 `data.y_original_*`、
> `conf/rl.conf` 的 `reward.freq_*` 与 `num_points`，以及代理模型的 `out_dim`
> （= 3 × 频点数）。
