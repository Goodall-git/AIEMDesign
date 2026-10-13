"""
本模块将常用的损失函数注册到全局注册表 LOSS，以便在项目其它地方通过注册表按名称获取和构造损失函数实例。
"""

import torch.nn as nn
from src.registry import LOSS

# 注册均方误差（MSE）损失：对小误差非常敏感，常用于回归且对离群点敏感的场景
LOSS.register(nn.MSELoss)
LOSS.register(nn.L1Loss)

# 注册 Huber 损失（平滑 L1）：结合了 MSE 与 MAE 的优点，对离群点更鲁棒
# 在 PyTorch 中可以通过 nn.HuberLoss(delta=...) 指定 delta，
# delta 越小越偏向 MAE，越大越偏向 MSE —— 可作为超参数调优。
LOSS.register(nn.HuberLoss)