# AI4EMDesign 训练结果分析报告

- 分析日期：2026-10-14
- 分析对象：`surrogate_output/01/`（仿真代理模型）与 `rl_output/01/`（GRPO 搜索优化）
- 结论摘要：**代理模型在训练分布内拟合良好，但外推到「全填充」区域时失效；策略模型正是利用了这一点，把 reward 从真实数据的上限 ~67 刷到了 78.5，而 Top-K 设计全是代理模型的幻觉，不具备物理意义。**

---

## 1. 训练配置与产出概览

| 项目 | 仿真代理模型 | 搜索优化模型（RL） |
|---|---|---|
| 入口 | `src.surrogate_model_training.main` | `src.policy_model_training.main` |
| 配置 | `conf/surrogate.conf` | `conf/rl.conf` |
| 模型 | `EMResnet(layers=[2,2,2], in_ch=3, out=303)`，0.19M 参数 | `AntennaRowPolicyModel(3,19,15)`，4.26M 参数 |
| 训练量 | 90 epoch，117 iter/epoch | 10000 iter，batch=32 |
| 耗时 | 08:22:29 → 08:23:52（约 1.5 分钟） | 08:32:29 → 08:46:06（约 13.5 分钟） |
| 产物 | `checkpoint.pt` / `best_checkpoint.pt` / events / code.zip | 同左 + `topk_samples.md` |

两次训练都正常跑完，无报错（日志中未出现异常堆栈）。

---

## 2. 仿真代理模型分析

### 2.1 曲线（TensorBoard `surrogate_output/01`）

| epoch | train/loss | eval/loss |
|---:|---:|---:|
| 1 | 9.19 | 7.40 |
| 10 | 3.15 | 3.39 |
| 20 | 1.99 | 3.24 |
| 40 | 1.28 | 3.37 |
| 58 | 0.92 | **2.80（最优）** |
| 70 | 0.78 | 2.86 |
| 90 | **0.67** | 2.82 |

- 验证损失在第 58 个 epoch 达到最优 2.80 dB，之后不再下降；训练损失却一路降到 0.67 dB。
- **最终 eval/train ≈ 4.2 倍**，说明模型已经明显过拟合，epoch 58 之后是纯粹的记忆化。
- 注意 `best_checkpoint.pt` 保存的是 epoch 58 的权重，而 `checkpoint.pt` 是最后一轮（epoch 90）的权重。

### 2.2 逐 S 参数误差

| 曲线 | 平均 L1(dB) | 最差点 |
|---|---:|---:|
| S11 | 2.57 | 15.2 dB @ 0 GHz |
| S21 | **3.67** | 37.7 dB @ 0 GHz |
| S22 | 2.56 | 15.2 dB @ 0 GHz |

- **低频端（0～1 GHz）误差最大**，S21 最差；高频段（>3 GHz）拟合较好，整体误差降到 ~1 dB。
- 这与物理直觉一致：低频端 S 参数变化剧烈，且数据里 S21 可低至 -60 dB（截断下界），相对误差大。

### 2.3 关键问题：分布内准，分布外崩

我在验证集上直接对比了「代理模型预测 reward」与「真实 reward」：

| 子集 | 真实 reward 均值 | 预测 reward 均值 | MAE |
|---|---:|---:|---:|
| 全部训练集（3762 个真实设计） | 3.13 | 3.98 | 0.87 |
| 真实 reward Top-10% | 47.4 | 40.7 | **9.2** |
| 真实 reward Top-1%（训练集） | 60.4 | 50.0 | **10.4** |

- 全样本相关系数高达 0.996，但**越是高 reward 的样本，代理模型低估得越厉害**（Top-1% 低估 10 dB）。
- 原因：训练集中根本不存在「高填充率」样本——训练数据每层填充率均值仅 `[0.446, 0.506, 0.024]`，单样本最高填充率 0.554，99 分位仅 0.465。模型从未见过填充率 > 0.6 的版图。

**外推测试（用 RL 实际加载的 `checkpoint.pt`）**：

| 输入 | 预测 reward |
|---|---:|
| 随机版图，填充率 0.70 | 59.6 |
| 随机版图，填充率 0.90 | 79.9 |
| 随机版图，填充率 0.99 | 92.0 |
| 全 1 版图 | **94.3** |
| 全 0 版图 | -33.5 |

- 代理模型的预测 reward **随填充率单调上升**，且在训练分布之外（填充率 0.7～1.0）依然线性外推，给出越来越离谱的高 reward。
- 对照：真实数据里 `corr(填充率, 真实 reward) ≈ -0.03`（几乎无关），真实数据最高 reward 只有 **67.15**，Top-1% 均值 60.4。**没有任何真实设计能达到 77 以上。**

---

## 3. 搜索优化模型（GRPO）分析

### 3.1 训练曲线（`rl_output/01`）

| iter 区间 | raw reward | 每步 entropy(nats) | \|adv\| | raw policy_loss |
|---|---:|---:|---:|---:|
| 0–200 | 43.24 | 10.40（=均匀分布上限 ln32768） | 2.2e-7 | -0.60 |
| 3000–3200 | 47.03 | 10.27 | 3.6e-7 | -1.26 |
| 5000–5200 | 50.07 | 10.06 | 4.9e-7 | -2.40 |
| 7000–7200 | 75.76 | 0.91 | 1.8e-6 | -10.82 |
| 9800–10000 | **77.12** | **0.36** | 5.4e-6 | -1.01 |

- reward 从 43.2 涨到 77.1，看起来提升巨大，但**增长集中在 iter≈6600 之后**（rolling-100 达到最终值 90% 的位置是 iter 6627）。
- 同期 entropy 从 10.40 崩到 0.36 nats（相当于每步动作概率分布极度尖锐），是典型的**策略塌缩 / 过早收敛**：前半程 6600 次迭代几乎在原地踏步（reward 43→50），后半程突然锁死到一个局部模式。
- 训练前期 |advantage| ~1e-7、policy_loss ~-0.6，梯度信号极弱，说明 GRPO 的 group-relative 归一化在 32 个样本的 group 上被数值噪声主导。

### 3.2 Top-K 结果不可信

`topk_samples.md`（n_samples=100，top_k=10）里 10 个设计的 reward 全部挤在 **77.78～78.51**，且互相之间只差 12～22 个 bit（855 bit 中 1.4%～2.6%），本质是**同一个解**。它们共同的特征是：

| 指标 | Top-K 设计 | 真实训练数据 |
|---|---:|---:|
| 整体填充率 | 0.91～0.93 | 均值 0.318，最高 0.554，99 分位 0.465 |
| V1 层填充率 | **0.933** | **0.024** |
| TM3 层填充率 | 0.933 | 0.506 |
| 预测 reward | 77.8～78.5 | 真实上限 67.15 |

- Top-K 的预测 S11/S22 ≈ -40 dB、S21 ≈ -1.7 dB，全都落在**代理模型外推出来的虚假区域**。
- V1 层在真实数据里几乎是空的（填充率 2.4%），策略却把它填成 93%——因为代理模型对 V1 过度敏感（把 V1 从全 0 改成全 1，平均会让预测 reward 变化 60.5 dB），策略顺着这个虚假梯度一路爬到了全 1。
- 换句话说：**reward 78.5 是「奖励劫持 / reward hacking」，不是真的找到了更好的天线。**

### 3.3 日志指标本身有 bug（影响判读）

1. **scheduler 从未生效**：`train/lr` 全程恒为 0.0001。原因见 `src/policy_model_training/engine.py`——`scheduler.step()` 在 `writer.add_scalar("train/lr", lr, iter_)` 之后调用，而 `lr` 是循环外一次性读取的常量，`CosineAnnealingLR` 完全没起作用。
2. **`train/reward` 等「avg」指标是全局累计平均**，从不按 log 间隔重置（只有 `time_cost_metric` / `speed_tester` 被 reset）。所以最后打印的 `avg_reward=57.46` 严重滞后于真实的 `raw_reward=77.14`；`avg_entropy=389` 也是 57 步求和后的量级。**看曲线时应以 `raw_*` 为准。**
3. **`train/adv` 恒为 ~0**：GRPO 的 advantage 做了 z-score，均值天然为 0，这条曲线没有信息量，应改成记录 `adv.std()`。
4. **断点恢复加载的是 `checkpoint.pt` 而非 `best_checkpoint.pt`**，且 `StateCheckPoint.restore` 只判断文件是否存在就直接 `load_state_dict`，不做形状校验——之前 `rl_output/01` 里残留旧 `row_bits` 的 checkpoint 时，就直接抛了 `size mismatch` 崩溃。

---

## 4. 根因总结

```
数据覆盖窄（填充率 ≤0.55）
        │
        ▼
代理模型在分布外无约束外推（reward 随填充率单调↑，全 1 给 94）
        │
        ▼
RL 在冻结代理模型上无约束最大化 reward
        │
        ▼
策略塌缩到「高填充率」的分布外区域，刷出 78.5 的虚假高分
```

核心矛盾：**代理模型只在填充率 ≤0.55 的窄分布上被监督过，却被当作全空间可用的奖励函数；而 RL 恰好会去攻击它没见过的区域。** 这不是 RL 训练本身出错，而是奖励模型的可信域问题。

---

## 5. 改进建议

### P0 —— 先让 reward 可信（最关键）

1. **给代理模型的输入分布加约束**。最简单有效的做法是在训练集里补入高填充率样本；若拿不到，可在代理模型训练时对输入做 dropout / 随机翻转做数据增强，并**专门在「外推区」做验证**（例如构造一批填充率 0.6～1.0 的输入，检查预测是否仍然单调）。
2. **训练一个「不确定性」信号并在 reward 里惩罚它**。例如用 ensemble / MC-dropout 估计代理模型的预测方差，`reward_final = reward_pred - λ * uncertainty`，让 RL 不敢往没见过的区域跑。
3. **给 reward 加物理/分布惩罚项**：对偏离训练分布的版图（按填充率、按每层统计）加正则；或直接限制策略动作的可行域（如每行最多 k 个 1）。
4. **校准 top 区域**：既然代理模型在 Top-1% 上低估 10 dB，可以考虑对高 reward 区域加权训练，或改用能抑制外推的损失（如带先验的 Huber / 分位数损失）。

### P1 —— 修 RL 训练本身

5. **修 scheduler**：把 `lr = optimizer.param_groups[0]["lr"]` 移进循环，或在 `scheduler.step()` 后再读 lr 并写入 TensorBoard。
6. **修 metrics**：`reward_metric` / `entropy_metric` / `policy_loss_metric` 在每次 log 后 `reset()`，或干脆只保留 `raw_*`；`train/adv` 改记 `adv.std()`。
7. **缓解策略塌缩**：把 `entropy_coef` 从 0.001 调大（如 0.01～0.05），或设 `entropy_coef` 下限 / 目标 entropy；提高采样温度；把 group size（batch_size=32）调大，让 GRPO 的 group-relative 估计更稳。
8. **topk 导出时提高采样量并加多样性约束**：`n_samples=100` 太少，10 个结果几乎是同一个解，建议上千次采样，并对 Top-K 做去重（如汉明距离阈值）。
9. **导出可信性指标**：在 `topk_samples.md` 里同时给出「该设计距训练分布的距离（填充率、最近邻距离）」和「代理模型不确定度」，方便人工判断哪些结果是可信的。

### P2 —— 工程健壮性

10. `StateCheckPoint.restore` 增加形状校验：不匹配时告警并从零训练，而不是直接崩。
11. 续训时默认加载 `best_checkpoint.pt`（或显式配置加载哪个）。
12. 复现实验前先清理 / 更换 `-o` 输出目录，避免残留 checkpoint 干扰（本次 `rl_output/01` 就踩过这个坑）。

---

## 6. 复现方式

```bash
# 代理模型
python -m src.surrogate_model_training.main --conf conf/surrogate.conf -o surrogate_output/02

# RL（务必确认 load_from 指向本次代理模型的 best_checkpoint）
python -m src.policy_model_training.main --conf conf/rl.conf -o rl_output/02 \
  -M surrogate_model.load_from=surrogate_output/02/best_checkpoint.pt

# 查看曲线
tensorboard --logdir=surrogate_output/   # 或 rl_output/
```

> 判断「结果是否可信」的最小检查：把 Top-K 设计喂给代理模型，看其填充率是否落在训练分布内；若填充率远超训练集（如本项目的 0.9+ vs 0.55 上限），基本可以判定为 reward hacking。
