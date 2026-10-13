# AI4EMDesign

## 项目简介

AI4EMDesign 是一个专注于智能电磁设计的代码库，旨在通过机器学习和强化学习技术优化电磁设备的设计。该项目适合电信学院的学生学习和研究，帮助他们理解如何将人工智能技术应用于电磁领域。

整体流程分两阶段：

1. **仿真代理模型（surrogate）**：监督学习，学习「三层版图 → S 参数曲线」的映射，充当快速仿真器；
2. **搜索优化模型（policy）**：强化学习（GRPO），在冻结的代理模型上搜索更优的三层版图。

## 目录结构

```
AI4EMDesign/
├── conf/                          # 配置文件目录
│   ├── surrogate.conf             # 仿真模型训练配置
│   └── rl.conf                    # 强化学习训练配置
├── data/                          # 数据目录
│   ├── raw_data/                  # 原始数据
│   │   ├── npz/                   # 三层二值矩阵 TM2 / TM3 / V1
│   │   ├── csv/                   # S11 / S21 / S22 的幅度(dB)
│   │   └── gds/                   # 版图文件（训练流程不使用）
│   ├── build_dataset.py           # 数据转换脚本（npz + csv → parquet）
│   ├── train.parquet              # 训练集（由脚本生成）
│   ├── val.parquet                # 验证集（由脚本生成）
│   └── README.md                  # 数据说明文档
├── src/                           # 源代码目录
│   ├── surrogate_model_training/  # 仿真模型训练模块
│   ├── policy_model_training/     # 强化学习训练模块
│   ├── models/                    # 模型定义（EMResnet / AntennaRowPolicyModel）
│   ├── data/                      # 数据处理模块
│   ├── reward/                    # 奖励函数模块
│   └── torchutils/                # PyTorch 工具模块
├── tests/                         # 单元测试
└── README.md
```

## 环境配置

1. Python 3.8 或更高版本
2. PyTorch 和 torchvision（请参考 [PyTorch 官网](https://pytorch.org) 安装）
3. 其他依赖：

```bash
pip install pyhocon tensorboard pandas pyarrow
```

## 关键尺寸（重要）

| 项目 | 形状 / 取值 |
|---|---|
| 输入 x（`topology`） | `(3, 19, 15)`，通道顺序 `TM2, TM3, V1`，取值 0/1 |
| 输出 y（`s_mag_db`） | `(3, 101)`，顺序 `S11, S21, S22`，单位 dB |
| 频率轴 | 0.0 ~ 10.0 GHz，101 点（步长 0.1 GHz） |
| dB 截断区间 | `[-60, 0]` |
| 代理模型 | `EMResnet(layers=[2,2,2], in_channels=3, out_dim=303)` |
| 策略模型 | `AntennaRowPolicyModel(n_matrices=3, n_rows=19, row_bits=15)`，57 步，词表 32768 |
| 数据集划分 | 3762 训练 / 418 验证（seed=42，9:1） |

## 数据准备

原始数据已包含在 `data/raw_data/` 下，需要先转换成 Parquet：

```bash
python data/build_dataset.py
```

脚本会读取 4180 组 `npz` / `csv` 配对，把频率统一重采样到 101 点、dB 截断到
`[-60, 0]`，再按 9:1 划分并写出 `data/train.parquet` 与 `data/val.parquet`。

详细格式说明请参考 `data/README.md`。

## 仿真代理模型训练

```bash
python -m src.surrogate_model_training.main --conf conf/surrogate.conf -o surrogate_output/01
```

训练结果（TensorBoard 事件、断点、代码快照）保存在 `surrogate_output/01/` 下。

## 搜索优化模型：强化学习训练

1. 确认 `conf/rl.conf` 中 `surrogate_model.load_from` 指向上一步的输出
   （默认 `surrogate_output/01/checkpoint.pt`）。
2. 运行：

```bash
python -m src.policy_model_training.main --conf conf/rl.conf -o rl_output/01 \
  -M surrogate_model.load_from=surrogate_output/01/checkpoint.pt
```

训练结果保存在 `rl_output/01/` 下。

## 结果分析

### TensorBoard

```bash
tensorboard --logdir=surrogate_output/   # 仿真模型结果
tensorboard --logdir=rl_output/          # 强化学习结果
```

打开浏览器访问 `http://localhost:6006`（TensorBoard 默认端口）。

代理模型记录的曲线：

- `train/lr`、`train/loss`、`train/l1_distance`
- `eval/lr`、`eval/loss`、`eval/l1_distance`
- `eval/l1_distance/s11`、`eval/l1_distance/s21`、`eval/l1_distance/s22`
  —— 逐频点的 L1 误差，横轴是真实频率(GHz)，用来定位哪一段频带拟合得不好

强化学习记录的曲线：

- `train/lr`、`train/policy_loss`、`train/reward`、`train/entropy`、`train/adv`
- 以及对应的 `train/raw_*` 单批原始值

### Top-K 设计导出

强化学习训练结束后会自动把采样结果的前 K 名写成
`rl_output/01/topk_samples.md`，每个样本包含：

- `TM2` / `TM3` / `V1` 三个 19×15 拓扑矩阵
- 指定频段（默认 6.4 ~ 7.1 GHz）上 S11 / S21 / S22 的预测幅度(dB)
- reward 数值

## 奖励函数（RL 优化目标）

在指定频段内，S11、S22 越小越好，S21 越大越好：

```
reward = -w11 * mean(S11_dB) - w22 * mean(S22_dB) + w21 * mean(S21_dB)
```

由于 dB 为负值，该式恰好等价于「S11/S22 越负越好、S21 越接近 0 dB 越好」。
频段与权重在 `conf/rl.conf` 的 `reward` 段配置。
