# 版本说明（version.md）

## V3.0（代理模型适配当前数据集 + 抑制过拟合）

**目标**：让代理模型适配现有数据集（3762 / 418 条，3×19×15 版图 → 3×101 dB 曲线），
并解决 V2.0 暴露的「训练轮次过多、严重过拟合」问题。

**结果（val L1 / dB，越低越好）**

| | train L1 | val L1 | 泛化差距 | 备注 |
|---|---:|---:|---:|---|
| V2.0（旧配置，epoch 59） | 0.31 | **2.61** | 8.6× | epoch 59 后 31 个 epoch 无改善 |
| V3.0（本次） | 0.40 | **1.50** | 3.9× | 验证最优出现在 epoch 125 |

实测方式：8 卡并行跑了 6 轮共 ~120 组超参（容量 / dropout / weight decay / 学习率 /
频率卷积层数 / 无源性惩罚 / EMA / 数据增强），每组用 2~3 个随机种子确认稳定性。

### 改动清单

| 文件 | 改动 |
|---|---|
| `conf/surrogate.conf` | 重写：`max_epochs` 90→200、`head_dropout` 0→0.25、`weight_decay` 1e-5→2e-3、`lr` 1e-3→5e-3、`head_conv_layers` 3→1；新增 `early_stop_patience` / `early_stop_min_delta` / `grad_clip_norm` / `ema_decay`；`loss` 换成 `PassivityPenalizedLoss` |
| `src/loss/passivity.py` | **新增**：无源性软约束损失（`PassivityPenalizedLoss`、`passivity_penalty`） |
| `src/loss/__init__.py` | 导入并注册 `PassivityPenalizedLoss` |
| `src/surrogate_model_training/main.py` | 新增早停、EMA、梯度裁剪；保存时用 EMA 权重；日志同时打印 val 与 best；修复 `set_reproducible` 判断的是函数对象（恒真）的老 bug |
| `src/surrogate_model_training/engine.py` | 支持 `grad_clip_norm` / `ema`；新增 `ModelEMA`；逐频点 L1 采集不再只在 `epoch//10==0` 时进行（原条件疑似 `% 10` 写错） |
| `src/torchutils/common.py` | `StateCheckPoint.restore` 增加形状/键校验，不匹配时告警并从零训练，而不是直接抛异常中断 |
| `conf/rl.conf` | `surrogate_model` 段同步新结构（`head_conv_layers: 1`、`head_dropout: 0.25`）；`load_from` 指向 `surrogate_output/03/best_checkpoint.pt` |
| `tests/test_passivity_loss.py` | **新增**：无源性惩罚与嵌套 criterion 构造的单测 |
| `README.md` | 更新回归头配置示例与调参建议 |

### 关键超参与依据

1. **dropout 0.25（最大单项收益）**：`head_dropout` 作用在「展平 → 投影」那一层，
   它才是过拟合的主要来源。0 → 0.2 就能把 val L1 从 2.49 降到 2.28。
2. **weight_decay 2e-3**：5M 参数 / 3762 样本，旧值 1e-5 形同虚设。
   dropout + wd 组合后 val L1 进一步降到 ~2.0。
3. **lr 5e-3 + 余弦退火**：小模型 + 强正则下，更大的步长收敛更快也更稳。
   同一组正则下 lr=1e-3 / 2e-3 / 3e-3 / 5e-3 的 val L1 约为 2.0 / 1.94 / 1.75 / 1.5。
4. **`head_conv_layers` 3 → 1**：数据里的曲线由 4~8 个真实频点插值而来，
   更深的频率卷积只是在拟合插值伪影。收益不大但稳定（同配置下 1 层比 3 层
   约好 0.05~0.2 dB，且参数量更少），故取 1 层。
5. **无源性软约束 λ=0.05**：见下方「诚实说明」。
6. **早停 + `max_epochs` 与 `T_max` 一致**：旧版 90 epoch 中后 31 个 epoch 是纯浪费；
   但 `max_epochs` 同时是余弦退火的 `T_max`，若两者不一致（例如 T_max=300 却早停在 163），
   学习率还没退到低位就停，val 会明显偏差（实测 1.72 vs 1.50）。

### 诚实说明：无源性约束只解决了一部分

`report.md` 建议把无源性作为损失项。本次实现了，但实测结论是**它主要起正则作用，
不是硬约束**：

| λ | val L1（2 seed 均值） | 预测越界样本占比 |
|---:|---:|---:|
| 0（纯 L1） | ~1.51 | ~95% |
| 0.05 | **~1.40** | ~95% |
| 0.1 | ~1.56 | ~82% |
| 0.3 | ~1.48 | ~83% |

原因：真值本身贴着约束边界（带内 `|S11|^2+|S21|^2` 均值 0.988），dB 空间里
零点几 dB 的误差就足以越界，平方软惩罚无法彻底消除。要**保证**满足约束，
需要在输出层做参数化（例如令 `S11 = cosθ·f`、`S21 = sinθ·f` 之类的三角参数化），
这属于后续工作，本次未做。

同理，本次**没有**解决 `report.md` 第 13 节指出的两个根本问题：
- 训练集里**没有任何带通滤波器样本**（带内高 + 带外低 = 0 条），
  代理模型仍然无法学到「通带通、阻带阻」；
- 代理模型在训练分布外（高填充率）仍然会外推。

因此 **V3.0 的产物只是「更准、更不过拟合的回归器」，不能直接当作可信的 reward**。
RL 侧仍应按 `report.md` 第 13 节继续加联合密度约束 / 不确定性惩罚 / hinge 化 reward。

### 复现命令

```bash
# 训练代理模型（约 4 分钟）
python -m src.surrogate_model_training.main --conf conf/surrogate.conf -o surrogate_output/03

# 查看曲线
tensorboard --logdir=surrogate_output/
```

---

## V1.0

本次全部改动统一记为 **V1.0**，涵盖数据格式、优化目标、奖励函数、模型结构与
训练/推理代码。下面按改动点逐条说明。

### 改动总览

| # | 改动点 | 主要文件 |
|---|---|---|
| 1 | 数据转换与频率轴统一 | `data/build_dataset.py`（新增）、`data/README.md`（新增） |
| 2 | 优化目标与模型尺寸：三条 S 参数曲线 | `src/data/parquet_dataset.py`、`conf/surrogate.conf` |
| 3 | 策略模型改为逐行自回归（57 步 × 15 bit） | `src/models/policy.py` |
| 4 | 奖励函数改为全频带滤波器形状 | `src/reward/band_rew.py`、`src/reward/__init__.py` |
| 5 | 代理模型回归头改造 | `src/models/resnet.py` |
| 6 | 训练与推理代码适配 | `src/policy_model_training/engine.py`、`infer.py`、`src/surrogate_model_training/main.py` |
| 7 | 配置更新 | `conf/surrogate.conf`、`conf/rl.conf` |
| 8 | 测试与文档更新 | `tests/test_parquet_dataset.py`、`README.md`、`version.md` |

### 关键尺寸对照

| 项目 | 改造前 | V1.0 |
|---|---|---|
| 输入 x | `(1, H, W)`（单矩阵） | `(3, 19, 15)`，通道顺序 `TM2, TM3, V1` |
| 输出 y | 单条曲线，1000 点 | `(3, 101)`，顺序 `S11, S21, S22`，单位 **dB** |
| 频率轴 | 随样本变化 | 统一 **0.0 ~ 10.0 GHz，101 点**（步长 0.1 GHz） |
| y 取值范围 | — | 截断到 **[-60, 0] dB** |
| 代理模型 | `in_channels=1, out_dim=1000` | `in_channels=3, out_dim=303` |
| 代理模型回归头 | GAP + 单层 Linear | 保留空间位置 + 频率轴 1D 卷积 |
| 策略模型 | 4 步 × 4 bit，词表 16 | 57 步 × 15 bit，词表 **32768** |
| 奖励函数 | 只看带内三参数均值 | 全频带（带内 + 带外滚降） |
| 数据集划分 | — | 3762 训练 / 418 验证（seed=42，9:1） |
| 通带 | — | **6.4 ~ 7.1 GHz**（0.1 GHz 栅格上共 8 个点，索引 64~71） |

---

### 1. 数据转换与频率轴统一

**问题**：原始 CSV 的频率轴不统一——4180 个样本共 **554 种不同网格**，
点数在 69 ~ 582 之间。而三个 S 参数又必须拼成定长向量才能做回归。

**做法**：新增 `data/build_dataset.py`，读取
`data/raw_data/npz/dataN.npz`（`TM2` / `TM3` / `V1`，各 `(19,15)` uint8）与
`data/raw_data/csv/dataN.csv`（`freq_ghz` + `S11/S21/S22_mag_db`），输出：

- `data/train.parquet`（3762 行）、`data/val.parquet`（418 行）
- 列 `topology`（展平 855 维 int8）、`s_mag_db`（展平 303 维 float32）
- schema metadata 写入形状、dtype、通道顺序、频率轴、dB 截断区间，供数据集读取

处理规则：

1. **频率重采样**：把每个样本的曲线线性插值到统一的 0 ~ 10 GHz / 101 点；
2. **dB 截断到 [-60, 0]**：原始 `S21_mag_db` 最低可到 -260 dB（数值噪声），
   不截断会让 L1 损失被离群值主导；
3. **划分**：`np.random.default_rng(42)` 打乱后 9:1 切分。

`data/raw_data/gds/` 下的版图文件训练流程不使用。

**涉及文件**：`data/build_dataset.py`（新增）、`data/README.md`（新增）

---

### 2. 优化目标与模型尺寸：三条 S 参数曲线

**问题**：优化目标从"单条曲线"变成"S11/S21/S22 三条曲线的幅值"，
输入也从单个矩阵变成三个矩阵，整条数据链路和模型尺寸都要跟着改。

**做法**：

- `src/data/parquet_dataset.py`：x 重建为 `(3, 19, 15)`、y 重建为 `(3, 101)`
  后展平成 `(3N,)`；下采样改为沿最后一维切片 `x[..., ::factor]`；
  新增频率轴解析（`dataset.freq_ghz`）与 Parquet 元数据读取。
- 代理模型：`in_channels: 1 → 3`、`out_dim: 1000 → 303`。
  骨架 `CifarResNet` 本身不用改（`padding=1` + `AdaptiveAvgPool` 对任意尺寸成立）。

**涉及文件**：`src/data/parquet_dataset.py`、`conf/surrogate.conf`

> 注意：`y_enable_stat_normalize` 必须保持 `false`，否则 y 不再是 dB，
> 奖励函数的物理含义会被破坏；`s_curve_downsample_factor` 必须保持 `1`，
> 否则奖励的频段索引会错位。

---

### 3. 策略模型：逐行自回归（57 步 × 15 bit）

**问题**：原来一次生成 4 bit × 4 步，对应单个 4×4 矩阵；现在要生成
三个 19×15 的矩阵，词表和步数都要放大。

**做法**：`src/models/policy.py` 改为 `AntennaRowPolicyModel`：

- 逐行生成：3 个矩阵 × 19 行 = **57 步**，每步输出一行 **15 bit**；
- 词表 **2^15 = 32768**；
- 返回 `actions (B,57)`、`topo_bits (B,3,19,15)`、`logp_sum (B,)`、`entropy_sum (B,)`。

**涉及文件**：`src/models/policy.py`

---

### 4. 奖励函数：全频带滤波器形状

**问题**：最初的奖励只看带内（6.4 ~ 7.1 GHz）三个 S 参数的平均值：

```
reward = -w11 * mean(S11_dB) - w22 * mean(S22_dB) + w21 * mean(S21_dB)
```

它完全不约束带外，因此"带内通、带外也全通"的版图同样能拿高分——做不出
滤波器形状。要做带通滤波器，必须把**整个频率轴**纳入目标。

**做法**：`src/reward/band_rew.py` 新增 `BandpassFilterReward`。

| 区域 | 频段 | 要求 |
|---|---|---|
| 带内（通带） | 6.4 ~ 7.1 GHz（8 点） | S21 越大越好（插损小）；S11、S22 越小越好（匹配好） |
| 带外（阻带） | 其余 93 个频点 | S21 越小越好（抑制 / 滚降，形成滤波形状的关键） |

```
reward = + w_s21_pass * agg(S21_dB[带内])
         - w_s11_pass * agg(S11_dB[带内])
         - w_s22_pass * agg(S22_dB[带内])
         - w_s21_stop * agg(S21_dB[带外])
```

- 带内三项对频点取**平均**；
- 带外那一项由 `stopband_aggregation` 决定：`mean` 取带外平均抑制水平，
  `max` 取带外**最差**的那一点（更严格，逼所有带外频点都压下去）。

因为 dB 是负值，各项符号自然成立，无需额外取反：

- S11/S22 带内越负（越小）→ `-mean` 越大 → reward 越高 ✓
- S21 带内越接近 0（越大）→ `+mean` 越大 → reward 越高 ✓
- S21 带外越负（越小）→ `-mean` 越大 → reward 越高 ✓

#### 4.1 频段索引必须用整数运算

`round((f - f_start) / step)`，**不能用浮点比较**：
`np.linspace(0, 10, 101)` 的第 71 个点实际是 `7.1000000000000005`，
用 `<= 7.1` 判断会把 7.1 GHz 这个点漏掉（6.4~7.1 GHz 会只剩 7 个点）。

#### 4.2 ⚠️ 权重失衡会导致 reward hacking

带内有 3 项、带外有 1 项都奖励"越负越好"，**只有带内 S21 一项**奖励
"越大越好"。如果让前者总和过大，策略会去找"整条曲线一起压到 -60 dB"的
退化解（例如整块金属屏蔽），因为那样能白拿三项分，而不是去做真正的滤波器。

用默认权重 `w_s21_pass=2.0, w_s11_pass=w_s22_pass=w_s21_stop=1.0` 估算
（dB 截断到 -60，属量级估算）：

| 设计 | S21 带内 | S11 带内 | S22 带内 | S21 带外 | reward |
|---|---|---|---|---|---|
| 退化解（整条压到 -60） | -60 | -60 | -60 | -60 | `-120+60+60+60 = +60` |
| 典型滤波器 | -1 | -20 | -20 | -40 | `-2+20+20+40 = +78` |

真实滤波器胜出，说明**带内传输权重取 2.0 是必要的**：若把它降回 1.0，
两者会变成 `+60` 对 `+59`，退化解反而略占优。

由于这类估算依赖实际 dB 水平，不够稳妥，另提供 `target_*_db` 阈值：
设值后对应项变成**只罚不奖**的 hinge 惩罚，达到指标后不再加分，
从机制上彻底消除刷分动机：

```hocon
target_s21_pass_db: -3.0    # 带内插损不低于 -3 dB
target_s11_pass_db: -10.0   # 带内回波不高于 -10 dB
target_s22_pass_db: -10.0
target_s21_stop_db: -20.0   # 带外抑制至少 20 dB
```

此时退化解与合格设计的奖励都趋于 0（或只剩违规惩罚），策略失去追求
无意义 -60 dB 的动机。

> 代价：若一个 batch 内所有设计都满足指标，奖励全为 0，
> GRPO 的组内优势 `(r - r.mean()) / r.std` 也全为 0，梯度消失。
> 这属于「已达标」，可以接受。

#### 4.3 关于 reward 尺度

GRPO 用的是组内相对优势 `adv = (r - r.mean()) / (r.std + eps)`，
所以 reward 的**绝对大小不影响策略梯度**，更换奖励函数不需要重新调学习率。
（但 `entropy_coef` 是相对 `logp_sum` 尺度的，见「需要留意的超参数」。）

#### 4.4 对照实验

想比较"只看带内"与"全频带滤波器形状"，只需改一行 `type_`：

```hocon
reward {
    type_: MultiSParamBandReward   # 只优化带内（旧）
    # type_: BandpassFilterReward  # 全频带 + 滚降（新，默认）
    ...
}
```

注意两个类的配置键不同：旧的用 `bands` / `weight_s11` / `weight_s21`，
新的用 `passband` / `weight_s11_pass` / `weight_s21_pass` / `weight_s21_stop` 等。
两个类都保留在代码里。

**涉及文件**：`src/reward/band_rew.py`、`src/reward/__init__.py`

---

### 5. 代理模型回归头改造

**问题**：改造前的代理模型尾部是：

```python
self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
self.fc = nn.Linear(64, out_dim)
```

这带来两个对本任务不利的问题：

1. **全局平均池化抹掉空间位置。** 它把整个 19×15 版图平均成 64 个数。
   而电磁响应强依赖金属图形**在哪里**——同一个图形平移到不同位置，S 参数
   完全不同。卷积本身是平移等变的，经过 GAP 后变成平移**不变**，与物理相悖；
   网络只能靠通道去隐式编码位置，代价很高。
2. **303 个输出彼此独立。** 它们是同一个 64 维向量的 303 个线性组合，
   S11/S21/S22 之间、相邻频点之间的相关性完全没有建模，也没有频率平滑先验，
   预测曲线容易出现锯齿。

**做法**：`src/models/resnet.py` 中新增两个回归头，通过 `head_type` 切换。

#### 5.1 `GAPHead`（旧结构，保留用于对照）

等价于改造前的 `avgpool + fc`，参数名变为 `head.avgpool.*` / `head.fc.*`。
保留它的目的是能一键切回旧结构做 A/B 对照实验。

#### 5.2 `SpectralHead`（新结构，默认）

```python
# 1) 保留全部空间位置：C*H*W -> hidden
self.fc = nn.Linear(in_channels * H * W, hidden)

# 2) 频率位置编码：让每个频点拥有自己的偏置
self.freq_pos = nn.Parameter(torch.zeros(1, hidden, n_freq))

# 3) 频率方向的 1D 卷积：局部平滑 + 相邻频点共享权重
self.refine = nn.Sequential(
    Conv1d(hidden, mid, k=5, padding=2), BatchNorm1d(mid), ReLU(),
    Conv1d(mid,    mid, k=5, padding=2), BatchNorm1d(mid), ReLU(),
    Conv1d(mid,    mid, k=5, padding=2), BatchNorm1d(mid), ReLU(),
)

# 4) 输出 (B, n_s_params, n_freq)
self.out_conv = nn.Conv1d(mid, n_s_params, kernel_size=1)
```

前向流程：

```
feat (B, 64, 19, 15)
  -> flatten(1)                      (B, 18240)       ← 285 个空间位置一个都不丢
  -> Linear + ReLU (+ Dropout)       (B, hidden)
  -> unsqueeze(-1) + freq_pos        (B, hidden, 101) ← 频率轴由此建立
  -> refine（3 层 1D 卷积）           (B, mid, 101)    ← 相邻频点共享卷积核 ⇒ 频率平滑先验
  -> out_conv（1x1）                  (B, 3, 101)
  -> flatten(1)                      (B, 303)         ← 接口保持不变
```

**输出仍按 `(n_s_params, n_freq)` 行优先展平成 `(B, out_dim)`**，因此下游完全
不需要改：`L1Loss` 依旧对展平的 303 维目标计算，奖励函数与 `infer.py` 依旧
`reshape(B, 3, 101)`，顺序天然一致。

#### 5.3 其他配套调整

- `CifarResNet.__init__` / `EMResnet` 工厂新增参数：`head_type`、`spatial`、
  `n_freq`、`head_hidden`、`head_mid`、`head_conv_layers`、`head_conv_kernel`、
  `head_dropout`。
- 权重初始化循环从只覆盖 `nn.Conv2d` / `nn.BatchNorm2d` 扩展到同时覆盖
  `nn.Conv1d` / `nn.BatchNorm1d`（新头里的 1D 卷积同样用 kaiming 初始化）。
- `SpectralHead.forward` 增加空间尺寸校验：若输入特征图与 `spatial` 不匹配，
  直接抛 `ValueError` 并提示检查 `conf` 里的 `model.spatial`，避免只看到
  `Linear` 的形状错误。
- `out_dim` 必须是 `n_freq` 的整数倍，否则构造时即报错。

#### 5.4 参数量变化（预估，未实跑验证）

| | 主干 | 回归头 | 合计 |
|---|---|---|---|
| `head_type: gap`（旧） | 约 0.17M | 约 0.02M | **约 0.19M** |
| `head_type: spectral`（新） | 约 0.17M | 约 5.02M | **约 5.20M** |

新头的主要开销来自 `fc: 64*19*15 -> 256`（18240×256 ≈ 4.67M）。
如果显存或训练速度吃紧，**优先调小 `head_hidden`**（例如 256 → 128 可让该层
减半），其次可以调小 `head_mid`。

#### 5.5 自测

`src/models/resnet.py` 底部带有自测块，会分别打印两种头的输出形状和参数量：

```bash
python src/models/resnet.py
```

预期输出形如 `spectral head: output=(2, 303), params=5.20M` 和
`gap head: output=(2, 303), params=0.19M`。

> 注意：**不要**用 `python -m src.models.resnet` 运行。`-m` 会先导入
> `src.models` 包（此时已注册 `EMResnet`），再执行一次本文件，触发
> `KeyError: 'EMResnet is already registered in model'`。

#### 5.6 兼容性

- **已训练的 checkpoint 全部失效。** 回归头由 `avgpool + fc` 变为 `head.*`，
  参数名已改变，旧权重无法加载。`surrogate_output/01/checkpoint.pt` 以及
  `rl_output/01/` 下的策略模型都需要重新训练。
- 该 checkpoint 本来就与新数据（旧数据为单矩阵输入、1000 点输出）不兼容，
  因此这次改动没有额外损失。
- 对外接口未变：模型仍返回 `(B, out_dim)`，`infer.py`、奖励函数、训练引擎
  均无需修改。

**涉及文件**：`src/models/resnet.py`

---

### 6. 训练与推理代码适配

| 文件 | 改动 |
|---|---|
| `src/policy_model_training/engine.py` | 策略输出已是 `(B,3,19,15)`，去掉多余的 `unsqueeze(1)` |
| `src/surrogate_model_training/main.py` | 逐频点 L1 误差按 S11/S21/S22 拆成三条曲线，横轴改用真实频率(GHz) |
| `src/policy_model_training/infer.py` | 适配三矩阵 / 三曲线；Top-K 导出与 TensorBoard 写入 |

代理模型现在记录的曲线：

- `train/lr`、`train/loss`、`train/l1_distance`
- `eval/lr`、`eval/loss`、`eval/l1_distance`
- `eval/l1_distance/s11`、`eval/l1_distance/s21`、`eval/l1_distance/s22`
  —— 逐频点 L1 误差，横轴是真实频率，用来定位哪一段频带拟合得不好

---

### 7. 配置更新

`conf/surrogate.conf` 的 `model` 段与 `conf/rl.conf` 的 `surrogate_model` 段
都新增了回归头配置：

```hocon
head_type: spectral     # spectral(新，默认) | gap(旧，对照用)
spatial: [19, 15]       # 输入版图尺寸，必须与数据一致
n_freq: 101             # 频率点数；out_dim 必须等于 n_s_params * n_freq
head_hidden: 256        # 展平后的投影维度（参数量主要来自这里）
head_mid: 128           # 频率方向 1D 卷积的通道数
head_conv_layers: 3
head_conv_kernel: 5
head_dropout: 0.0
```

> ⚠️ **两处配置必须完全一致。** `conf/rl.conf` 的 `surrogate_model` 段用于重建
> 代理模型并加载 `checkpoint.pt`，若与 `conf/surrogate.conf` 的 `model` 段
> 不一致，`load_state_dict` 会因参数名/形状不匹配直接报错（strict 模式）。

`conf/rl.conf` 的 `reward` 段改为：

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

---

### 8. 测试与文档更新

| 文件 | 改动 |
|---|---|
| `tests/test_parquet_dataset.py` | 断言更新到新形状（`(3,19,15)` 输入、`(3N,)` 输出） |
| `README.md` | 新增「快速开始」「代理模型的回归头」章节；关键尺寸表与奖励函数章节重写 |
| `version.md` | 本文档 |

---

## 沿用未改的已知问题

以下几项是改造前就存在的问题，为控制改动范围本次**未修改**，在此记录备查：

1. **`set_reproducible` 开关失效。** `main.py` 中 `if set_reproducible:` 判断的是
   导入的函数对象而非配置项，条件恒为真，导致该配置项被忽略，
   `torch.backends.cudnn.benchmark` 分支永远走不到。
2. **逐频点 L1 曲线只在 epoch 1~9 采集。** 采集条件是 `if epoch // 10 == 0`
   （疑似本想写 `% 10`），且横轴是频率，后续 epoch 会覆盖同一批频点，
   因此最终显示的是 epoch 9 的误差剖面而非历史。
3. **`loss` 与 `l1_distance` 当前完全相同。** 配置里 criterion 是 `L1Loss`，
   而 `l1_distance = |outputs - targets|.mean()` 与它算的是同一个量。
   只有把 criterion 换成 `MSELoss` / `HuberLoss`，这两条曲线才会分开。
4. **验证集指标按 batch 平均而非按样本平均。** `AverageMetric` 累加每个 batch
   的值再除以 batch 个数。418 条验证数据 = 13 个满 batch + 1 个 2 条的 batch，
   那 2 条样本占了 1/14 权重而非 2/418，`eval/*` 会有轻微偏差。
5. **`src/config/__init__.py` 的正则用了非 raw 字符串**（`'\d'`），
   Python 3.12+ 会给出 `SyntaxWarning: invalid escape sequence '\d'`，
   功能正常，改成 `r'...'` 即可消除。

## 需要留意的超参数

1. **`tanh_constant` 已设为 `null`。** 原值 1.5 会把 logits 限制在 ±1.5，
   在词表 32768 下 softmax 接近均匀分布且梯度饱和，策略学不动。
   要回到旧行为设为 `1.5`。
2. **`entropy_coef: 0.001` 相对偏弱。** `logp_sum` / `entropy_sum` 是在 57 步上
   求和的（原来 4 步），熵正则相对于策略项的权重下降约 14 倍。
   想保持原相对强度建议提到 `0.01 ~ 0.05`。损失函数本身未改动。
3. **奖励权重需要留意平衡。** 默认权重（`weight_s21_pass: 2.0`，其余 1.0）
   是为了让真实滤波器优于"整条压到 -60 dB"的退化解，见第 4.2 节。
   如果改了权重，建议重新做一次那个估算，或者直接打开 `target_*_db`
   阈值来从机制上避免刷分。
4. **-60 dB 截断会在低频堆积，并影响带外项。** 接近 DC 处 S21 基本是隔离
   （原始值低至 -260 dB），截断后大量落在 -60 dB 地板上。奖励的带外项奖励
   "S21 越负越好"，而低频段天然就在地板上，所以带外项会有一大块"白送"的分，
   真正体现滚降质量的其实是**靠近通带边缘**的那一段。
   如果想突出滚降，把 `stopband_aggregation` 设为 `max`——带外最差点通常
   就出现在通带边缘尚未滚下去的位置，正好对应滚降能力。
5. **`hidden_size: 64` 对 32768 词表偏小**，可尝试 128 / 256。

## 后续命令

```bash
# 1) 生成数据（已生成过可跳过）
python data/build_dataset.py

# 2) 训练代理模型（V1.0 后必须重新训练）
python -m src.surrogate_model_training.main --conf conf/surrogate.conf -o surrogate_output/01

# 3) 强化学习搜索
python -m src.policy_model_training.main --conf conf/rl.conf -o rl_output/01 \
  -M surrogate_model.load_from=surrogate_output/01/checkpoint.pt

# 4) 结果查看
tensorboard --logdir=surrogate_output/     # 或 rl_output/
```

代理模型改造后第一次训练，建议先确认模型输出的确是 `(B, 303)`、
且 `Model Parameters` 约为 5.2M（若仍显示 0.19M，说明配置没生效）。

## V3.0（2026-10-18）匹配网络

- 任务从「带通滤波器」改为「匹配网络」：带内需同时存在 S11/S22 深极小值。
- 频段由 8.4–9.3 GHz 改为 **7.5–9.9 GHz**：真实凹陷样本 24 → 136（训练）、4 → 14（验证）。
- `ResonanceWeightedLoss` 新增显式凹陷深度、位置、虚假凹陷抑制三项监督。
- `SpectralHead.freq_mod_rank` 8 → 24，`head_conv_kernel` 5 → 3。
- 策略预训练新增 `pretrain.min_hamming` 多样性筛选（默认 0 = 关闭）。
- `MatchingNetworkReward` 新增 `max_dip_bonus_db`，给凹陷奖励封顶。
- 新增 `conf/surrogate_resonance_v3.conf`、`conf/rl_match_wide.conf`、
  `tests/test_resonance_loss.py`、`tests/test_matching_reward.py`。
- 产物：`surrogate_output/11_resonance_wideband`、`rl_output/14_match_wide`。
- Top-10：S11 凹陷 6.6–10.8 dB、S22 5.4–6.9 dB，填充率 0.43–0.46（分布内）。
