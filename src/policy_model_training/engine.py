import logging
import torch
import torch.nn as nn
import torch.optim as optim

from src.torchutils.distributed import world_size
from src.torchutils.metrics import AverageMetric, EstimatedTimeArrival
from src.torchutils.common import ThroughputTester, time_enumerate
from src.torchutils.common import StateCheckPoint
from src.torchutils.common import MetricsStore
from torch.utils.tensorboard import SummaryWriter

_logger = logging.getLogger(__name__)


def run_one_epoch_grpo(
    policy_model: nn.Module,
    surrogate_model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler | None,
    device: str,
    log_interval: int,
    save_interval: int,
    # RL configs
    max_episodes: int,
    batch_size: int,
    reward_fn,                       # callable(pred) -> (B,)
    entropy_coef: float,
    adv_eps: float,
    grad_clip_norm: float | None,
    #
    writer: SummaryWriter,
    metric_store: MetricsStore,
    state_ckpt: StateCheckPoint,
    states: dict,
):
    """
    纯强化学习 GRPO 训练（无 DataLoader、无 eval）

    一个 epoch 包含 n_iters 次 policy 更新；
    每次更新用 batch_size 个 topology 作为一个 group。
    """

    # policy：永远是 train
    policy_model.train()

    # surrogate：永远冻结
    surrogate_model.eval()
    for p in surrogate_model.parameters():
        p.requires_grad_(False)

    # metrics
    time_cost_metric = AverageMetric("time_cost")
    policy_loss_metric = AverageMetric("avg_policy_loss", format_digits=8)
    reward_metric = AverageMetric("avg_reward", format_digits=8)
    entropy_metric = AverageMetric("avg_entropy", format_digits=8)
    adv_metric = AverageMetric("avg_adv", format_digits=8)

    eta = EstimatedTimeArrival(max_episodes)
    speed_tester = ThroughputTester()

    lr = optimizer.param_groups[0]["lr"]
    _logger.info(
        f"RL TRAIN start, epoch={max_episodes:04d}, lr={lr:.6f}, "
        f"iters={max_episodes}, batch={batch_size}"
    )

    for time_cost, iter_, _ in time_enumerate(range(max_episodes), start=1):

        # -------------------------------------------------
        # 1) policy 采样 topology（一个 group）
        # -------------------------------------------------
        actions, topo_bits, logp_sum, entropy_sum = policy_model(
            batch_size=batch_size
        )

        topo_bits = topo_bits.to(device)
        logp_sum = logp_sum.to(device)
        entropy_sum = entropy_sum.to(device)

        # (B, n_matrices, n_rows, row_bits) -> (B, 3, 19, 15)，0/1 映射到 -1/+1
        # 与监督训练时 x_enable_scale_to_minus1_1 的处理保持一致
        topo = topo_bits.float() * 2.0 - 1.0
        # import ipdb; ipdb.set_trace()

        # -------------------------------------------------
        # 2) surrogate forward → reward
        # -------------------------------------------------
        with torch.no_grad():
            pred = surrogate_model(topo)
            rewards = reward_fn(pred)
            # import ipdb; ipdb.set_trace()

        # -------------------------------------------------
        # 3) GRPO advantage（group-relative）
        # -------------------------------------------------
        r_mean = rewards.mean()
        r_std = rewards.std(unbiased=False)
        adv = (rewards - r_mean) / (r_std + adv_eps)

        # -------------------------------------------------
        # 4) policy loss（REINFORCE + GRPO）
        # -------------------------------------------------
        policy_loss = -(adv.detach() * logp_sum).mean()

        if entropy_coef != 0.0:
            policy_loss -= entropy_coef * entropy_sum.mean()

        # -------------------------------------------------
        # 5) update policy
        # -------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        policy_loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(), grad_clip_norm
            )

        optimizer.step()

        # -------------------------------------------------
        # 6) metrics / log
        # -------------------------------------------------
        time_cost_metric.update(time_cost)
        policy_loss_metric.update(policy_loss)
        reward_metric.update(rewards.mean())
        entropy_metric.update(entropy_sum.mean())
        adv_metric.update(adv.mean())

        eta.step()
        speed_tester.update(topo)

        if iter_ % log_interval == 0 or iter_ == max_episodes:
            _logger.info(", ".join([
                "RL-TRAIN",
                f"iter={iter_:05d}/{max_episodes:05d}",
                f"time={time_cost_metric.compute()*1000:.2f}ms",
                f"fps={speed_tester.compute()*world_size():.0f} samples/s",
                f"raw policy_loss={policy_loss.item():.8f}",
                f"{policy_loss_metric}",
                f"raw reward={rewards.mean().item():.8f}",
                f"{reward_metric}",
                f"raw entropy={entropy_sum.mean().item():.8f}",
                f"{entropy_metric}",
                f"raw adv={adv.mean().item():.8f}",
                f"{adv_metric}",
                f"{eta}",
            ]))
            time_cost_metric.reset()
            speed_tester.reset()
            
        writer.add_scalar("train/lr", lr, iter_)
        writer.add_scalar("train/raw_policy_loss", policy_loss.item(), iter_)
        writer.add_scalar("train/policy_loss", policy_loss_metric.compute(), iter_)
        writer.add_scalar("train/raw_reward", rewards.mean().item(), iter_)
        writer.add_scalar("train/reward", reward_metric.compute(), iter_)
        writer.add_scalar("train/raw_entropy", entropy_sum.mean().item(), iter_)
        writer.add_scalar("train/entropy", entropy_metric.compute(), iter_)
        writer.add_scalar("train/raw_adv", adv.mean().item(), iter_)
        writer.add_scalar("train/adv", adv_metric.compute(), iter_)


        if scheduler is not None:
            scheduler.step()
            

        
        metric_store += {
            "train/lr": lr,
            "train/policy_loss": policy_loss_metric.compute(),
            "train/reward": reward_metric.compute(),
            "train/entropy": entropy_metric.compute(),
        }
        
        if iter_ % save_interval == 0 or iter_ == max_episodes:
            state_ckpt.save(metric_store=metric_store, states=states)



