# -*- coding: utf-8 -*-
"""
电磁代理模型的物理约束损失。

二端口无源网络的散射矩阵必须满足能量守恒：

    |S11|^2 + |S21|^2 <= 1
    |S21|^2 + |S22|^2 <= 1

代理模型只被 dB 曲线监督，没有任何机制保证这一点。实测（见 report.md 第 8.2 节）
发现旧模型有 94% 的预测样本违反该约束，而这些「物理上不可能」的预测恰好
落在高 reward 区域，RL 会直接冲进去 —— 这正是 reward hacking 的物理成因。

本模块把违反量作为软惩罚加到回归损失上，让模型在拟合 dB 曲线的同时
尽量落在物理可行域内。惩罚项写成

    relu(|S11|^2 + |S21|^2 - 1)^2 + relu(|S21|^2 + |S22|^2 - 1)^2

（平方而非一次方，是为了让轻微越界只受很小的推力，避免把拟合精度整体拉低。）
"""

import torch
import torch.nn as nn

from src.registry import LOSS

# 代理模型输出按 [S11(N), S21(N), S22(N)] 的行优先顺序展平，
# 与 data/build_dataset.py 的 s_param_order 一致。
N_S_PARAMS = 3


def s_db_to_amplitude(s_db: torch.Tensor) -> torch.Tensor:
    """dB -> 线性幅度。"""
    return torch.pow(10.0, s_db / 20.0)


def passivity_violation(
    s_db: torch.Tensor,
    n_freq: int,
    n_s_params: int = N_S_PARAMS,
) -> torch.Tensor:
    """
    计算每个频点的无源性越界量（已过 relu，非负）。

    Args:
        s_db: (B, n_s_params * n_freq) 的 dB 预测，行优先展平
        n_freq: 频率点数
        n_s_params: S 参数条数（当前数据为 3：S11/S21/S22）

    Returns:
        (B, n_freq) 的越界量，其中只包含「两个约束取较大者」的逐点最大值，
        便于统计违反率；若需要逐约束的惩罚请用 passivity_penalty。
    """
    if s_db.dim() != 2 or s_db.shape[1] != n_s_params * n_freq:
        raise ValueError(
            f"passivity_violation 期望 (B, {n_s_params * n_freq}) 的输入，"
            f"实际得到 {tuple(s_db.shape)}")
    amp = s_db_to_amplitude(s_db)
    s11, s21, s22 = amp.split(n_freq, dim=1)
    r1 = (s11 ** 2 + s21 ** 2 - 1.0).clamp(min=0.0)
    r2 = (s21 ** 2 + s22 ** 2 - 1.0).clamp(min=0.0)
    return torch.maximum(r1, r2)


def passivity_penalty(
    s_db: torch.Tensor,
    n_freq: int,
    n_s_params: int = N_S_PARAMS,
) -> torch.Tensor:
    """逐约束的平方惩罚（未加权），供 PassivityLoss 使用。"""
    amp = s_db_to_amplitude(s_db)
    s11, s21, s22 = amp.split(n_freq, dim=1)
    r1 = (s11 ** 2 + s21 ** 2 - 1.0).clamp(min=0.0)
    r2 = (s21 ** 2 + s22 ** 2 - 1.0).clamp(min=0.0)
    return (r1 ** 2 + r2 ** 2).mean()


@LOSS.register
class PassivityPenalizedLoss(nn.Module):
    """
    数据拟合损失 + 无源性软约束。

        loss = criterion(pred, target) + lambda_passivity * mean(relu(|S|^2 sum - 1)^2)

    在 conf 里用：

        loss {
            type_: PassivityPenalizedLoss
            criterion { type_: L1Loss }      # 或 MSELoss / HuberLoss
            n_freq: 101
            lambda_passivity: 1e-3
        }

    lambda_passivity = 0 时退化为纯 criterion，便于对照实验。
    """

    def __init__(self, criterion, n_freq: int, n_s_params: int = N_S_PARAMS,
                 lambda_passivity: float = 0.0):
        super(PassivityPenalizedLoss, self).__init__()
        # criterion 既可以直接传 nn.Module，也可以在 conf 里写嵌套的
        #   criterion { type_: L1Loss }
        # 这里把嵌套配置递归构造成模块。
        if isinstance(criterion, dict) or hasattr(criterion, "get") and hasattr(criterion, "keys"):
            from src.registry import LOSS as _LOSS
            criterion = _LOSS.build_from(criterion)
        if not isinstance(criterion, nn.Module):
            raise TypeError(
                f"criterion 需要是 nn.Module 或含 type_ 的配置，实际为 {type(criterion)}")
        self.criterion = criterion
        self.n_freq = int(n_freq)
        self.n_s_params = int(n_s_params)
        self.lambda_passivity = float(lambda_passivity)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(outputs, targets)
        if self.lambda_passivity <= 0:
            return loss
        return loss + self.lambda_passivity * passivity_penalty(
            outputs, n_freq=self.n_freq, n_s_params=self.n_s_params)
