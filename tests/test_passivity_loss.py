from pathlib import Path

import torch
import torch.nn as nn
from pyhocon import ConfigFactory

from src.registry import LOSS
import src.loss  # noqa: F401  触发注册

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SURROGATE_CONF = PROJECT_ROOT / "conf" / "surrogate.conf"


def test_passivity_penalty_zero_when_feasible():
    """满足 |S11|^2+|S21|^2<=1 与 |S21|^2+|S22|^2<=1 时惩罚为 0。"""
    from src.loss.passivity import passivity_penalty

    n_freq = 5
    # 三条曲线都远低于 0 dB，必定无源
    s = torch.full((2, 3 * n_freq), -20.0)
    assert passivity_penalty(s, n_freq=n_freq).item() == 0.0


def test_passivity_penalty_positive_when_violated():
    from src.loss.passivity import passivity_penalty

    n_freq = 4
    # 三条曲线都取 0 dB -> |S| = 1 -> 两个约束都越界 1.0
    s = torch.zeros(1, 3 * n_freq)
    penalty = passivity_penalty(s, n_freq=n_freq).item()
    assert penalty > 0.0


def test_passivity_penalized_loss_builds_from_conf():
    """conf 里的嵌套 criterion 配置应当被正确构造成模块。"""
    cfg = ConfigFactory.parse_file(str(SURROGATE_CONF)).get_config("loss")
    loss = LOSS.build_from(cfg)

    assert isinstance(loss, nn.Module)
    assert isinstance(loss.criterion, nn.L1Loss)
    assert loss.lambda_passivity > 0
    assert loss.n_freq == cfg.get_int("n_freq")

    # 与纯 L1 相比，越界输入应当得到更大的 loss
    n_freq = loss.n_freq
    targets = torch.full((1, 3 * n_freq), -30.0)
    outputs = torch.zeros(1, 3 * n_freq)  # 0 dB -> 严重越界
    l1_only = nn.L1Loss()(outputs, targets)
    assert loss(outputs, targets).item() > l1_only.item()


def test_passivity_penalized_loss_lambda_zero_is_pure_criterion():
    from src.loss.passivity import PassivityPenalizedLoss

    loss = PassivityPenalizedLoss(criterion=nn.L1Loss(), n_freq=4, lambda_passivity=0.0)
    outputs = torch.zeros(1, 12)
    targets = torch.full((1, 12), -30.0)
    assert torch.allclose(loss(outputs, targets), nn.L1Loss()(outputs, targets))
