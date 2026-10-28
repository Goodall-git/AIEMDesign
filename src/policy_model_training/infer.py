import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# 默认频率轴，与 data/build_dataset.py 及 conf/rl.conf 的 reward 段保持一致
DEFAULT_FREQ_START_GHZ = 0.0
DEFAULT_FREQ_END_GHZ = 10.0
DEFAULT_NUM_POINTS = 101


# 与 data/build_dataset.py 的 metadata 保持一致
MATRIX_NAMES: Tuple[str, ...] = ("TM2", "TM3", "V1")
S_PARAM_NAMES: Tuple[str, ...] = ("S11", "S21", "S22")


@dataclass
class TopKSample:
    reward: float
    topo_bits: torch.Tensor      # (3, 19, 15) 0/1 long
    band_s: torch.Tensor         # (3, N_band) float，单位 dB
    full_s: torch.Tensor         # (3, num_points) float，单位 dB


def _format_matrix_as_markdown(topo_2d: torch.Tensor) -> str:
    """
    把 2D tensor 格式化成 markdown 代码块，便于阅读。
    """
    topo_cpu = topo_2d.detach().cpu()
    lines = []
    for r in topo_cpu:
        lines.append(" ".join(str(int(x)) for x in r.tolist()))
    return "```text\n" + "\n".join(lines) + "\n```"


def _format_vector_as_markdown(vec_1d: torch.Tensor, max_items: Optional[int] = None) -> str:
    """
    把 1D tensor 格式化成 markdown 代码块（vector 很长时可截断显示）。
    """
    v = vec_1d.detach().cpu().flatten()
    if max_items is not None and v.numel() > max_items:
        head = v[:max_items].tolist()
        text = ", ".join(f"{x:.4f}" for x in head) + f", ... (total={v.numel()})"
    else:
        text = ", ".join(f"{x:.4f}" for x in v.tolist())
    return "[" + text + "]"


def _format_curves_as_markdown(curves_2d: torch.Tensor,
                               names: Tuple[str, ...],
                               max_items: Optional[int] = None) -> str:
    """
    把 (3, N) 的三条曲线格式化成带 S11/S21/S22 标签的代码块。
    """
    lines = []
    for i, name in enumerate(names):
        lines.append(f"{name}: " + _format_vector_as_markdown(curves_2d[i], max_items=max_items))
    return "```text\n" + "\n".join(lines) + "\n```"


def _get_freq_axis(num_points: int, reward_fn) -> np.ndarray:
    """
    获取频率轴（GHz）。优先从 reward_fn.freqs 读取，保证与 reward 用的网格一致；
    否则回退到默认的 0.0 ~ 10.0 GHz 均匀网格。
    """
    freqs = getattr(reward_fn, "freqs", None)
    if torch.is_tensor(freqs) and freqs.numel() == num_points:
        return freqs.detach().cpu().numpy().astype(np.float64)
    f_start = float(getattr(reward_fn, "freq_start", DEFAULT_FREQ_START_GHZ))
    f_end = float(getattr(reward_fn, "freq_end", DEFAULT_FREQ_END_GHZ))
    return np.linspace(f_start, f_end, num_points, dtype=np.float64)


@torch.no_grad()
def sample_topk_policy_samples(
    policy_model: nn.Module,
    surrogate_model: nn.Module,
    reward_fn,
    device: str,
    # sampling config
    n_samples: int = 100,
    batch_size: int = 16,
    top_k: int = 10,
    # topology -> surrogate config（与训练一致）
    topology_scale_m11: bool = True,       # True: topo*2-1, False: 0/1
) -> List[TopKSample]:
    """
    采样 n_samples 个三层拓扑，评估 reward，返回按 reward 降序排列的 Top-K 样本。

    这是纯推理的核心逻辑，被 markdown 导出与 TensorBoard 导出共用。
    每个 TopKSample 内含完整的 (3, num_points) S 参数曲线（单位 dB）。
    """
    policy_model.eval()
    surrogate_model.eval()
    for p in surrogate_model.parameters():
        p.requires_grad_(False)

    if not hasattr(reward_fn, "num_points"):
        raise ValueError("reward_fn 必须提供 num_points（MultiSParamBandReward 已提供）")
    num_points = int(reward_fn.num_points)

    collected: List[TopKSample] = []
    remaining = n_samples

    while remaining > 0:
        bs = min(batch_size, remaining)

        # policy 采样：(B, 3, 19, 15)
        actions, topo_bits, logp_sum, entropy_sum = policy_model(batch_size=bs)
        topo_bits = topo_bits.to(device)

        topo = topo_bits.float()
        if topology_scale_m11:
            topo = topo * 2.0 - 1.0

        # surrogate 预测：(B, 3 * num_points)
        pred_s = surrogate_model(topo)

        if pred_s.ndim == 2 and pred_s.shape[1] == 3 * num_points:
            pred_s = pred_s.reshape(pred_s.shape[0], 3, num_points)
        if pred_s.ndim != 3 or pred_s.shape[1] != 3 or pred_s.shape[2] != num_points:
            raise ValueError(
                f"surrogate_model 输出应为 (B, {3 * num_points}) 或 "
                f"(B, 3, {num_points})，当前 pred_s.shape={tuple(pred_s.shape)}"
            )

        rewards = reward_fn(pred_s)         # (B,)
        if rewards.ndim != 1 or rewards.shape[0] != bs:
            raise ValueError(f"reward_fn 输出应为 (B,)，当前 rewards.shape={tuple(rewards.shape)}")

        for i in range(bs):
            collected.append(
                TopKSample(
                    reward=float(rewards[i].item()),
                    topo_bits=topo_bits[i].detach().cpu(),   # 存 CPU，避免占 GPU
                    band_s=torch.empty(0),                   # 频段曲线由调用方按需截取
                    full_s=pred_s[i].detach().cpu(),         # (3, num_points)
                )
            )

        remaining -= bs

    collected.sort(key=lambda x: x.reward, reverse=True)
    return collected[:top_k]


def export_topk_policy_samples_to_tensorboard(
    top_samples: List[TopKSample],
    log_dir,
    reward_fn,
    tag_prefix: str = "topk_design",
):
    """
    把 Top-K 设计的**完整 S 参数曲线**写入 TensorBoard。

    每个设计写成一个独立的 run（子目录），这样在 TensorBoard 里
    S11 / S21 / S22 三张图中各会出现 K 条曲线，可直接对比。

    写入的 tag（每个 run 下）：
    - {tag_prefix}/S11, .../S21, .../S22 : 逐频点幅度(dB)，横轴=频率(MHz)
    - {tag_prefix}/reward                 : 该设计的 reward（常量曲线）

    实现要点：先在内存中构建一个临时 SummaryWriter，再把事件文件拷贝到目标
    run 子目录，从而绕开 "同一 SummaryWriter 只能对应一个 run" 的限制。

    注意：TensorBoard 的 scalar step 必须是整数，不能直接传 GHz 浮点数
    （7.5/7.6/... 会被截断成 7/7/...，曲线会出现阶梯）。这里统一用 MHz
    作为 step，例如 7500 表示 7.5 GHz。
    """
    import glob
    import shutil
    import tempfile

    if len(top_samples) == 0:
        return

    num_points = int(getattr(reward_fn, "num_points", DEFAULT_NUM_POINTS))
    freq = _get_freq_axis(num_points, reward_fn)

    logdir = str(log_dir)
    os.makedirs(logdir, exist_ok=True)

    for rank, s in enumerate(top_samples, start=1):
        full_s = s.full_s.detach().cpu()
        if full_s.ndim != 2 or full_s.shape[0] != len(S_PARAM_NAMES):
            raise ValueError(
                f"TopKSample.full_s 形状应为 ({len(S_PARAM_NAMES)}, {num_points})，"
                f"当前为 {tuple(full_s.shape)}"
            )

        run_name = f"{tag_prefix}_rank{rank:02d}"
        run_dir = os.path.join(logdir, run_name)
        os.makedirs(run_dir, exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_writer = SummaryWriter(log_dir=tmp)
            for si, sname in enumerate(S_PARAM_NAMES):
                curve = full_s[si].tolist()
                for fi in range(num_points):
                    tmp_writer.add_scalar(
                        f"{tag_prefix}/{sname}", curve[fi],
                        int(round(float(freq[fi]) * 1000)))
            tmp_writer.add_scalar(f"{tag_prefix}/reward", s.reward, 0)
            tmp_writer.flush()
            tmp_writer.close()

            # 把临时事件文件搬到目标 run 目录（重命名以避免同名冲突）
            for ev in glob.glob(os.path.join(tmp, "events.out.tfevents.*")):
                shutil.copy(ev, os.path.join(run_dir, f"{run_name}.{os.path.basename(ev)}"))


def export_topk_policy_samples_to_markdown(
    policy_model: nn.Module,
    surrogate_model: nn.Module,
    reward_fn,
    device: str,
    out_md_path: str,
    # sampling config
    n_samples: int = 100,
    batch_size: int = 16,
    top_k: int = 10,
    # topology -> surrogate config（与训练一致）
    topology_scale_m11: bool = True,       # True: topo*2-1, False: 0/1
    # vector display
    max_band_points_to_print: Optional[int] = None,  # 例如 200，避免 markdown 过长；None 表示全打印
    # 复用外部已采样好的 Top-K（避免重复采样；None 表示内部重新采样）
    precomputed_samples: Optional[List[TopKSample]] = None,
):
    """
    纯推理：采样 n_samples 个三层拓扑，评估 reward，选 top_k 写入 markdown。

    Markdown 内容（每个样本）：
    - TM2 / TM3 / V1 三个拓扑矩阵
    - 指定频段上 S11 / S21 / S22 的预测幅度(dB)
    - reward 数值

    注意：
    - policy_model：建议 eval() 以减少随机层干扰（若有 dropout）
    - surrogate_model：eval() + no_grad()
    - 频段索引优先从 reward_fn 读取（reward_fn.band_idx）
    """
    os.makedirs(os.path.dirname(out_md_path) or ".", exist_ok=True)

    # ---------- 采样 Top-K（或复用外部结果） ----------
    if precomputed_samples is not None:
        top_samples = list(precomputed_samples)
    else:
        top_samples = sample_topk_policy_samples(
            policy_model=policy_model,
            surrogate_model=surrogate_model,
            reward_fn=reward_fn,
            device=device,
            n_samples=n_samples,
            batch_size=batch_size,
            top_k=top_k,
            topology_scale_m11=topology_scale_m11,
        )

    # ---------- 解析频段索引（用于 markdown 中的频段曲线） ----------
    if not hasattr(reward_fn, "band_idx"):
        raise ValueError("reward_fn 必须提供 band_idx（MultiSParamBandReward 已提供）")

    num_points = int(reward_fn.num_points)
    band_idx = reward_fn.band_idx
    if not torch.is_tensor(band_idx):
        band_idx = torch.as_tensor(band_idx, dtype=torch.long)
    band_idx_cpu = band_idx.detach().cpu()

    band_freqs = None
    if hasattr(reward_fn, "freqs") and torch.is_tensor(reward_fn.freqs):
        band_freqs = reward_fn.freqs.detach().cpu().flatten()[band_idx_cpu]

    # 用完整曲线截出频段部分，填充 band_s
    for s in top_samples:
        s.band_s = s.full_s[:, band_idx_cpu]

    # ---------- 写 Markdown ----------
    lines: List[str] = []
    lines.append("# Top-K Antenna Designs (Policy Inference)")
    lines.append("")
    lines.append(f"- Total samples evaluated: **{n_samples}**")
    lines.append(f"- Top-K selected: **{top_k}**")
    lines.append(f"- Matrix channels: **{', '.join(MATRIX_NAMES)}** (shape 19 x 15)")
    lines.append(f"- S-parameters (dB): **{', '.join(S_PARAM_NAMES)}**")
    lines.append("")

    # band 文案
    if hasattr(reward_fn, "band"):
        f0, f1 = reward_fn.band
        lines.append(f"- Band used for export: **{f0:.3f}–{f1:.3f} GHz**")
    lines.append(f"- Band grid: {band_idx.numel()} points"
                 + ("" if band_freqs is None else
                    " (" + ", ".join(f"{v:.3f}" for v in band_freqs.tolist()) + " GHz)"))
    lines.append("")

    for rank, s in enumerate(top_samples, start=1):
        lines.append(f"## #{rank} (reward = {s.reward:.8f})")
        lines.append("")
        for mi, mname in enumerate(MATRIX_NAMES):
            lines.append(f"### {mname}")
            lines.append(_format_matrix_as_markdown(s.topo_bits[mi]))
            lines.append("")
        lines.append(f"### Predicted S-parameters (band, dB)")
        lines.append(_format_curves_as_markdown(s.band_s, S_PARAM_NAMES,
                                               max_items=max_band_points_to_print))
        lines.append("")
        lines.append("---")
        lines.append("")

    with open(out_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return top_samples
