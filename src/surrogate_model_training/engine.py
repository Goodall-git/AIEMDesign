import copy
import functools
import logging

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.amp import autocast, GradScaler

from src.torchutils.distributed import world_size
from src.torchutils.metrics import AverageMetric, EstimatedTimeArrival
from src.torchutils.common import ThroughputTester, time_enumerate

_logger = logging.getLogger(__name__)


class ModelEMA:
    """
    权重指数滑动平均（Exponential Moving Average）。

    小数据集（本项目 3762 条）上模型的验证误差会随 epoch 剧烈波动，直接取
    「验证最优的那个 epoch」容易被噪声选到运气好的点。EMA 权重相当于把
    最近若干步的参数做平均，泛化通常更稳、更不容易过拟合。

    只在 train_one_epoch 里更新，eval 时用 ema_model 前向。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        for ema_p, p in zip(self.module.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # buffer（BatchNorm 的 running stats 等）直接复制，不做平均
        for ema_b, b in zip(self.module.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state_dict):
        self.module.load_state_dict(state_dict)

    def train(self, mode=True):
        self.module.train(mode)
        return self

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _run_one_epoch(is_training: bool,
                   epoch: int,
                   model: nn.Module,
                   loader: data.DataLoader,
                   criterion: nn.modules.loss._Loss,
                   optimizer: optim.Optimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   device: str,
                   log_interval: int,
                   grad_clip_norm: float = None,
                   ema: ModelEMA = None):
    phase = "train" if is_training else "eval"
    model.train(mode=is_training)

    time_cost_metric = AverageMetric("time_cost")
    loss_metric = AverageMetric("avg_loss", format_digits=8)
    l1_distance_metric = AverageMetric("avg_l1_distance", format_digits=8)
    eta = EstimatedTimeArrival(len(loader))
    speed_tester = ThroughputTester()

    lr = optimizer.param_groups[0]['lr']
    _logger.info(f"{phase.upper()} start, epoch={epoch:04d}, lr={lr:.6f}")

    point_wise_l1_distance_list = []

    for time_cost, iter_, (inputs, targets) in time_enumerate(loader, start=1):
        inputs = inputs.to(device=device, non_blocking=True)
        targets = targets.to(device=device, non_blocking=True)

        with torch.set_grad_enabled(mode=is_training):
            # 评估时如果启用了 EMA，用平均权重前向（这才是最终会保存/使用的权重）
            if not is_training and ema is not None:
                outputs = ema.module(inputs)
            else:
                outputs = model(inputs)
            loss: torch.Tensor = criterion(outputs, targets)

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            if grad_clip_norm is not None and grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)

        with torch.no_grad():
            l1_distance = torch.abs(outputs - targets).mean()

            if not is_training:
                point_wise_l1_distance = torch.abs(outputs - targets).mean(dim=0)
                point_wise_l1_distance_list.append(point_wise_l1_distance)

        time_cost_metric.update(time_cost)
        loss_metric.update(loss)
        l1_distance_metric.update(l1_distance)
        eta.step()
        speed_tester.update(inputs)

        if iter_ % log_interval == 0 or iter_ == len(loader):
            _logger.info(", ".join([
                phase.upper(),
                f"epoch={epoch:04d}",
                f"iter={iter_:05d}/{len(loader):05d}",
                f"fetch data time cost={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} images/s",
                f"raw loss={loss.item():.8f}",
                f"{loss_metric}",
                f"raw l1_distance={l1_distance.item():.8f}",
                f"{l1_distance_metric}",
                f"{eta}",
            ]))
            time_cost_metric.reset()
            speed_tester.reset()

    if is_training and scheduler is not None:
        scheduler.step()

    _logger.info(", ".join([
        phase.upper(),
        f"epoch={epoch:04d} {phase} complete",
        f"{loss_metric}",
        f"{l1_distance_metric}",
    ]))

    return {
        f"{phase}/lr": lr,
        f"{phase}/loss": loss_metric.compute(),
        f"{phase}/l1_distance": l1_distance_metric.compute(),
        f"{phase}/point_wise_l1_distance_list": point_wise_l1_distance_list
    }


train_one_epoch = functools.partial(_run_one_epoch, is_training=True)
evaluate_one_epoch = functools.partial(_run_one_epoch, is_training=False)
