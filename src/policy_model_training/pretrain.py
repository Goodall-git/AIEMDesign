# -*- coding: utf-8 -*-
"""
策略网络的监督预训练（behavior cloning）。

为什么需要
----------
本项目采用「逐行自回归 + 每步 32768 类」的动作空间。从均匀随机初始化
开始时，策略采样出的版图密度约为 0.50/0.50（训练集为 0.45/0.51），
且几乎不可能同时命中「深匹配」所需的精细结构，于是 GRPO 在很长一段时间
内 reward 方差很小、advantage 只有 1e-8 量级 —— 前 7000 iter 原地踏步
（report.md 第 3.1 / 11.1 节反复出现）。

解决办法是先用训练集里 reward 最高的那批版图做监督预训练，把策略的
初始分布推到「好设计附近」，再交给 RL 微调。这样 RL 从一开始就能拿到
非零的组内方差。

实现
----
只训练 action_embedding / start_token / lstm / logits_head，
损失是逐行交叉熵（teacher forcing）。数据里 3 个矩阵 × 19 行 = 57 步，
每一步的标签就是该行的 15 bit 整数编码。
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

_logger = logging.getLogger(__name__)


def _rows_to_actions(topo_bits: torch.Tensor) -> torch.Tensor:
    """
    topo_bits: (B, n_matrices, n_rows, row_bits) 的 0/1
    return:    (B, n_matrices * n_rows) 的行整数编码
    """
    b, m, r, bits = topo_bits.shape
    shifts = torch.arange(bits - 1, -1, -1, device=topo_bits.device, dtype=torch.long)
    weights = (1 << shifts).to(topo_bits.dtype)
    rows = (topo_bits.to(torch.long) * weights).sum(dim=-1)   # (B, m, r)
    return rows.reshape(b, m * r)


def pretrain_policy(policy_model: nn.Module,
                    topo_bits: torch.Tensor,
                    device,
                    epochs: int = 30,
                    batch_size: int = 64,
                    lr: float = 1e-3,
                    log_interval: int = 10) -> float:
    """
    用一组版图（0/1 张量，形状 (N, 3, 19, 15)）做监督预训练。

    返回最后一个 epoch 的平均交叉熵（nats/step），便于判断是否收敛。
    """
    if epochs <= 0 or topo_bits is None or topo_bits.numel() == 0:
        return float("nan")

    policy_model.train()
    actions = _rows_to_actions(topo_bits.to(device)).to(device)   # (N, 57)
    n = actions.shape[0]

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=lr, weight_decay=1e-4)

    _logger.info(f"[pretrain] {n} designs, {epochs} epochs, batch={batch_size}, lr={lr}")
    last_loss = float("nan")
    for ep in range(1, epochs + 1):
        perm = torch.randperm(n, device=device)
        total, count = 0.0, 0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if idx.numel() < 2:
                continue
            target = actions[idx]

            hx = policy_model._zeros(idx.numel())
            cx = policy_model._zeros(idx.numel())
            embed = policy_model.start_token.unsqueeze(0).expand(idx.numel(), -1)
            loss = 0.0
            for step in range(policy_model.n_steps):
                hx, cx = policy_model.lstm(embed, (hx, cx))
                logits = policy_model._scale_logits(policy_model.logits_head(hx))
                loss = loss + F.cross_entropy(logits, target[:, step])
                embed = policy_model.action_embedding(target[:, step])
            loss = loss / policy_model.n_steps

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()

            total += loss.item() * idx.numel()
            count += idx.numel()

        last_loss = total / max(count, 1)
        if log_interval and ep % log_interval == 0:
            _logger.info(f"[pretrain] epoch={ep:03d} ce={last_loss:.4f} nats/step")
    _logger.info(f"[pretrain] done, final ce={last_loss:.4f} nats/step")
    return last_loss
