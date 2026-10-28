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
│   │   ├── main.py                # 训练入口（含 Top-K 导出）
│   │   ├── infer.py               # 采样 / Top-K 导出（markdown 与 TensorBoard）
│   │   └── export_topk_to_tensorboard.py  # 对已有 RL 输出补写 Top-K 曲线
│   ├── models/                    # 模型定义（EMResnet / AntennaRowPolicyModel）
│   ├── data/                      # 数据处理模块
│   ├── reward/                    # 奖励函数模块
│   └── torchutils/                # PyTorch 工具模块
├── tests/                         # 单元测试
├── version.md                     # 改动记录（数据/目标改造、回归头改造）
└── README.md
```

## 环境配置

1. Python 3.8 或更高版本
2. PyTorch 和 torchvision（请参考 [PyTorch 官网](https://pytorch.org) 安装）
3. 其他依赖：

```bash
pip install pyhocon tensorboard pandas pyarrow
```

## 快速开始

从原始数据到最终结果，按顺序执行以下四步（在项目根目录下运行）：

```bash
# 1) 数据准备：npz + csv -> parquet（只需跑一次；仓库里已生成好可跳过）
python data/build_dataset.py

# 2) 训练仿真代理模型（学「版图 -> S 参数」的映射）
python -m src.surrogate_model_training.main --conf conf/surrogate.conf -o surrogate_output/01

# 3) 强化学习搜索更优版图（以冻结的代理模型为环境）
python -m src.policy_model_training.main --conf conf/rl.conf -o rl_output/01 \
  -M surrogate_model.load_from=surrogate_output/01/checkpoint.pt

# 4) 查看结果
tensorboard --logdir=surrogate_output/    # 代理模型
tensorboard --logdir=rl_output/           # 强化学习 + Top-K 设计曲线
```

然后打开浏览器访问 `http://localhost:6006`。奖励最高的前 K 个设计会同时写入
`rl_output/01/topk_samples.md` 和 TensorBoard（详见「结果分析」一节）。

> 第 3 步依赖第 2 步的 `checkpoint.pt`，必须等第 2 步训练完成后再跑。
> `conf/rl.conf` 里的 `surrogate_model.load_from` 默认已指向
> `surrogate_output/01/checkpoint.pt`，如果换过输出目录，用 `-M` 覆盖即可。

## 关键尺寸（重要）

| 项目 | 形状 / 取值 |
|---|---|
| 输入 x（`topology`） | `(3, 19, 15)`，通道顺序 `TM2, TM3, V1`，取值 0/1 |
| 输出 y（`s_mag_db`） | `(3, 101)`，顺序 `S11, S21, S22`，单位 dB |
| 频率轴 | 0.0 ~ 10.0 GHz，101 点（步长 0.1 GHz） |
| dB 截断区间 | `[-60, 0]` |
| 代理模型 | `EMResnet(layers=[2,2,2], in_channels=3, out_dim=303)` |
| 代理模型回归头 | `head_type: spectral`（默认，保留空间位置 + 频率轴 1D 卷积） |
| 策略模型 | `AntennaRowPolicyModel(n_matrices=3, n_rows=19, row_bits=15)`，57 步，词表 32768 |
| 数据集划分 | 3762 训练 / 418 验证（seed=42，9:1） |

> 模型尺寸与回归头的完整说明、以及改造前后的对比，见 `version.md`。

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

### 代理模型的回归头（`head_type`）

代理模型的骨架是 CIFAR 版 ResNet，**回归头**决定了「特征图 → 303 维 S 参数」这一段怎么走。
在 `conf/surrogate.conf` 的 `model` 段用 `head_type` 选择，共两种：

| `head_type` | 结构 | 适用场景 |
|---|---|---|
| `spectral`（**默认**） | 展平保留全部 19×15 空间位置 → `Linear` → 加频率位置编码 → 若干层 `Conv1d`（频率方向）→ `1x1 Conv1d` 出 3 通道 | 正常训练 |
| `gap` | `AdaptiveAvgPool2d((1,1))` + 单层 `Linear` | 只用于和旧结构做对照实验 |

前者的两个关键点：

1. **不做全局平均池化**。`AdaptiveAvgPool2d((1,1))` 会把整个 19×15 版图平均成 64 个数，
   空间位置信息全部丢失；而电磁响应强依赖金属图形**在哪里**。新头把 `C×H×W` 直接展平
   再投影，285 个空间位置一个都不丢。
2. **频率轴用 1D 卷积生成**。旧头里 303 个输出是同一个 64 维向量的 303 个独立线性组合，
   相邻频点之间没有任何约束；新头先在频率轴方向建立 `(B, hidden, 101)`，
   再用 3 层 `Conv1d`（核长 5）做局部建模，相邻频点共享卷积核，相当于加上了
   「S 参数曲线沿频率是光滑的」这一先验。

模型内部产出 `(B, 3, 101)`，最后仍按行优先展平成 `(B, 303)` 返回，
**所以对外接口没变**——`L1Loss`、奖励函数、`infer.py`、训练引擎都不需要改。

对应的配置项（`conf/surrogate.conf` 的 `model` 段）：

```hocon
model {
    type_: EMResnet
    layers: [2,2,2]
    in_channels: 3      # TM2 / TM3 / V1
    out_dim: 303        # 3 条 S 曲线 x 101 个频点

    head_type: spectral # spectral(默认) | gap(对照)
    spatial: [19, 15]   # 输入版图尺寸，必须与数据一致
    n_freq: 101         # 频率点数；out_dim 必须等于 n_s_params * n_freq
    head_hidden: 256    # 展平后的投影维度（参数量主要来自这里）
    head_mid: 128       # 频率方向 1D 卷积的通道数
    head_conv_layers: 1  # 实测 1 层优于 3 层（曲线由少量真实频点插值而来）
    head_conv_kernel: 5
    head_dropout: 0.25   # 旧版 0.0 会明显过拟合
}
```

> ⚠️ **`conf/rl.conf` 的 `surrogate_model` 段必须与上面完全一致。**
> RL 训练会按该配置重建代理模型再加载 `checkpoint.pt`，两边不一致会在
> `load_state_dict` 时报参数名/形状不匹配（strict 模式）。

#### 常见调整

| 想做的事 | 怎么改 |
|---|---|
| 显存/速度吃紧 | 调小 `head_hidden`（256 → 128 可让最大的一层减半） |
| 想换回旧结构做对照 | `head_type: gap`，**需要重新训练**（两种头的参数名不同，权重不能互载） |
| 改频率点数 | 同步改 `n_freq`、`out_dim`（= 3 × 点数）、`conf/rl.conf` 的 `reward.num_points`/`freq_*`，并重新生成数据 |
| 版图尺寸变了 | 改 `spatial`，否则 `SpectralHead` 会直接抛出提示检查 `model.spatial` 的 `ValueError` |
| 想加正则 | `head_dropout`（默认 0.25）。注意它只作用在展平投影那一层 |
| 想调训练轮次 | `max_epochs`（默认 200）**同时是余弦退火的 `T_max`**，两者必须一致 |
| 想提前停止 | `early_stop_patience`（默认 80，设 null 关闭） |

#### 自测

`src/models/resnet.py` 底部带有自测块，会分别打印两种头的输出形状和参数量：

```bash
python src/models/resnet.py
```

预期输出形如 `spectral head: output=(2, 303), params=5.20M` 和
`gap head: output=(2, 303), params=0.19M`。

> 注意：**不要**用 `python -m src.models.resnet` 运行。`-m` 会先导入 `src.models`
> 包（此时已注册 `EMResnet`），再执行一次本文件，触发
> `KeyError: 'EMResnet is already registered in model'`。

训练启动后，日志里的 `Model Parameters` 可以核对是否约 5.2M（`spectral`）。
若仍显示 0.19M，说明配置没生效。

## 搜索优化模型：强化学习训练

1. 确认 `conf/rl.conf` 中 `surrogate_model.load_from` 指向上一步的输出，
   并且**加载 `best_checkpoint.pt` 而不是 `checkpoint.pt`**（后者是最后一轮，
   往往是过拟合的权重）。
2. `conf/rl.conf` 的 `surrogate_model` 段必须与 `conf/surrogate.conf` 的 `model`
   段完全一致（含 `head_conv_layers` / `head_dropout`），否则 `load_state_dict` 会报错。
3. 运行：

```bash
python -m src.policy_model_training.main --conf conf/rl.conf -o rl_output/01 \
  -M surrogate_model.load_from=surrogate_output/03/best_checkpoint.pt
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
- `topk_design/S11`、`topk_design/S21`、`topk_design/S22`：训练结束后，把 reward
  最高的前 K 个版图的**完整 S 参数曲线**写入 TensorBoard。每个设计是一个独立
  run（`topk_design_rank01` ~ `topk_design_rank0K`，rank01 为 reward 最高），
  横轴为频率(GHz)，因此 S11/S21/S22 三张图里各能直接对比 K 条曲线；
  `topk_design/reward` 记录该设计的 reward 数值。

> 注意：这里的曲线是**代理模型的预测值**，可信度取决于代理模型在该输入分布上的
> 外推能力。若设计的填充率远超训练集分布（见 `report.md`），预测曲线可能失真。

### Top-K 设计导出

强化学习训练结束后会自动把采样结果的前 K 名写成
`rl_output/01/topk_samples.md`，同时把前 K 名版图的完整 S 参数曲线写入
TensorBoard（见上一节）。每个样本包含：

- `TM2` / `TM3` / `V1` 三个 19×15 拓扑矩阵
- 指定频段（默认 6.4 ~ 7.1 GHz）上 S11 / S21 / S22 的预测幅度(dB)
- reward 数值

### 对已有训练结果补写 Top-K 曲线

若想在不重新训练的情况下，为已有的 RL 输出目录补写（或改变 K）Top-K 曲线：

```bash
python -m src.policy_model_training.export_topk_to_tensorboard \
  --conf conf/rl.conf \
  --rl-output rl_output/01 \
  --policy-ckpt rl_output/01/checkpoint.pt \
  --top-k 5
```

脚本会重新采样 `infer.n_samples` 个版图、取前 K 名，写入
`<rl-output>/topk_design_rank0X/`，然后可用 `tensorboard --logdir=rl_output/` 查看。

## 奖励函数（RL 优化目标）

目标是让代理模型预测出的响应**具有带通滤波器的形状**，因此需要**全频带**
一起优化，而不只是看好带内。

### 目标

| 区域 | 频段 | 要求 |
|---|---|---|
| 带内（通带） | 6.4 ~ 7.1 GHz | S21 越大越好（插损小）；S11、S22 越小越好（匹配好） |
| 带外（阻带） | 其余全部频点 | S21 越小越好（抑制 / 滚降，这是形成滤波形状的关键） |

### 公式

```
reward = + w_s21_pass * agg(S21_dB[带内])
         - w_s11_pass * agg(S11_dB[带内])
         - w_s22_pass * agg(S22_dB[带内])
         - w_s21_stop * agg(S21_dB[带外])
```

因为 dB 是负值，各项符号自然成立：

- S11 / S22 带内越负（越小）→ `-mean` 越大 → reward 越高 ✓
- S21 带内越接近 0（越大）→ `+mean` 越大 → reward 越高 ✓
- S21 带外越负（越小）→ `-mean` 越大 → reward 越高 ✓

带内三项对频点取平均；带外那一项由 `stopband_aggregation` 决定：

- `mean`（默认）—— 带外平均抑制水平
- `max` —— 带外**最差**的那一点，更严格，逼所有带外频点都压下去

### 权重怎么理解

所有项都在同一个 dB 尺度上，所以权重就是**兑换比**。例如
`weight_s21_pass: 2.0` + `weight_s21_stop: 1.0` 表示
「带内 1 dB 插损 == 带外 2 dB 抑制」。

> ⚠️ **别让带外权重相对带内 S21 过大。** 带内有 3 项、带外有 1 项都奖励
> 「越负越好」，只有带内 S21 一项奖励「越大越好」。如果前者总和过大，
> 策略会去找「整条曲线一起压到 -60 dB」的退化解（例如整块金属屏蔽），
> 因为那样能白拿三项分，而不是真正的滤波器。默认把带内传输设为 2.0、
> 其余为 1.0，就是为了让真实滤波器优于这种退化解。

### 严格按指标设计（可选）

如果需要按滤波器指标来（例如「插损 < 3 dB、回波 < -10 dB、抑制 > 20 dB」），
把 `conf/rl.conf` 里的 `target_*_db` 设上，对应项会变成**只罚不奖**的
hinge 惩罚，达到指标后不再继续加分，从而彻底消除上面说的刷分动机：

```hocon
target_s21_pass_db: -3.0    # 带内插损不低于 -3 dB
target_s11_pass_db: -10.0   # 带内回波不高于 -10 dB
target_s22_pass_db: -10.0
target_s21_stop_db: -20.0   # 带外抑制至少 20 dB
```

> 注意此时若一个 batch 内所有设计都满足指标，奖励会全部为 0，
> GRPO 的组内优势也会全为 0（梯度消失）——这属于「已达标」，可接受。

### 配置

频段、权重、聚合方式、阈值都在 `conf/rl.conf` 的 `reward` 段：

```hocon
reward {
    type_: BandpassFilterReward
    passband: [6.4, 7.1]
    freq_start_ghz: 0.0
    freq_end_ghz: 10.0
    num_points: 101
    s_param_order: [S11, S21, S22]

    weight_s21_pass: 2.0
    weight_s11_pass: 1.0
    weight_s22_pass: 1.0
    weight_s21_stop: 1.0
    stopband_aggregation: mean

    target_s21_pass_db: null
    target_s11_pass_db: null
    target_s22_pass_db: null
    target_s21_stop_db: null

    clamp_db: [-60.0, 0.0]
}
```

> 代码里还保留了一个只优化带内、不约束带外的 `MultiSParamBandReward`，
> 把 `type_` 改回它即可做对照实验。

### 关于 reward 的尺度

GRPO 用的是组内相对优势 `adv = (r - r.mean()) / r.std`，所以 reward 的
**绝对大小不影响策略梯度**，换奖励函数不需要重新调学习率。
