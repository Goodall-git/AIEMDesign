import os
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any

import torch
import torch.nn as nn


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

    policy_model.eval()
    surrogate_model.eval()
    for p in surrogate_model.parameters():
        p.requires_grad_(False)

    # ---------- 解析频段索引与频率轴长度 ----------
    if not hasattr(reward_fn, "band_idx"):
        raise ValueError("reward_fn 必须提供 band_idx（MultiSParamBandReward 已提供）")
    if not hasattr(reward_fn, "num_points"):
        raise ValueError("reward_fn 必须提供 num_points（MultiSParamBandReward 已提供）")

    num_points = int(reward_fn.num_points)
    band_idx = reward_fn.band_idx
    if not torch.is_tensor(band_idx):
        band_idx = torch.as_tensor(band_idx, dtype=torch.long)
    band_idx = band_idx.to(device)

    band_freqs = None
    if hasattr(reward_fn, "freqs") and torch.is_tensor(reward_fn.freqs):
        band_freqs = reward_fn.freqs.detach().cpu().flatten()[band_idx.detach().cpu()]

    # ---------- 开始采样累计 ----------
    collected: List[TopKSample] = []
    remaining = n_samples

    with torch.no_grad():
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

            # reward
            rewards = reward_fn(pred_s)         # (B,)
            if rewards.ndim != 1 or rewards.shape[0] != bs:
                raise ValueError(f"reward_fn 输出应为 (B,)，当前 rewards.shape={tuple(rewards.shape)}")

            # 取频段曲线：(B, 3, N_band)
            band_s = pred_s[:, :, band_idx]

            # 存样本（逐个存，方便 TopK 排序）
            for i in range(bs):
                collected.append(
                    TopKSample(
                        reward=float(rewards[i].item()),
                        topo_bits=topo_bits[i].detach().cpu(),   # 存 CPU，避免占 GPU
                        band_s=band_s[i].detach().cpu(),
                        full_s=pred_s[i].detach().cpu(),
                    )
                )

            remaining -= bs

    # ---------- TopK ----------
    collected.sort(key=lambda x: x.reward, reverse=True)
    top_samples = collected[:top_k]

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
