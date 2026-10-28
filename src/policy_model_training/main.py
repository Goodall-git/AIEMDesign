import logging
import os
import dataclasses
import pprint

import torch
from torch import optim
import torch.cuda
import torch.utils.data
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.collect_env import get_pretty_env_info
from torch.utils.tensorboard import SummaryWriter
from pyhocon import ConfigTree

import src.models
import src.data
import src.loss
import src.optimizer
import src.scheduler
import src.reward

from src.config import Args, get_args
from src.registry import DATA, MODEL, OPTIMIZER, SCHEDULER, LOSS, REWARD
from src.policy_model_training.engine import run_one_epoch_grpo
from src.policy_model_training.pretrain import pretrain_policy
from src.policy_model_training.infer import (
    sample_topk_policy_samples,
    export_topk_policy_samples_to_markdown,
    export_topk_policy_samples_to_tensorboard,
)

from src.torchutils.common import set_cudnn_auto_tune, set_reproducible, generate_random_seed, disable_debug_api
from src.torchutils.common import set_proper_device, get_device
from src.torchutils.common import compute_nparam
from src.torchutils.common import StateCheckPoint
from src.torchutils.common import MetricsStore
from src.torchutils.metrics import EstimatedTimeArrival
from src.torchutils.logging import init_logger, create_code_snapshot


_logger = logging.getLogger(__name__)


def _load_pretrain_designs(conf: ConfigTree, surrogate_model, reward_impl, device):
    """
    从训练集里挑 reward 最高的 top_ratio 比例版图，作为策略的监督预训练数据。

    直接用真实 reward（而不是代理预测）排序，避免一开始就被代理模型的
    分布外幻觉带偏。
    """
    # rl.conf 里没有 data 段，直接按与 surrogate.conf 一致的约定读 parquet。
    import torch as _torch
    from src.data.parquet_dataset import (
        EMTopologySCurveParquetDataset, get_x_transforms, get_y_transforms,
    )
    root = conf.get_string("pretrain.data_root", "./data")
    filename = conf.get_string("pretrain.train_filename", "train.parquet")
    ds = EMTopologySCurveParquetDataset(
        os.path.join(root, filename), "topology", "s_mag_db",
        get_x_transforms(True), get_y_transforms(1))

    topo = _torch.stack([ds[i][0] for i in range(len(ds))]).to(device)   # (N,3,19,15), -1/1
    y = _torch.stack([ds[i][1] for i in range(len(ds))]).to(device)      # (N,303)

    with _torch.no_grad():
        rewards = reward_impl(y)

    top_ratio = conf.get_float("pretrain.top_ratio", 0.05)
    k = max(1, int(len(ds) * top_ratio))
    ranked = _torch.topk(rewards, k).indices

    # 可选：按 reward 绝对值再筛一道（min_reward，null/None 时关闭）。
    #
    # 为什么需要：当目标任务在数据集里覆盖稀疏时，top_ratio 选出的「前 k 个」
    # 里有大量 reward≈0 的样本 —— 它们只是「在所有候选里相对靠前」，并不真的
    # 满足任务。例如 7 GHz 任务：训练集 3762 个样本中真正在 6.5~7.5 GHz 形成
    # 带内深凹陷并拿到 reward>18 的只有 18 个，而 top_ratio=0.05 会取 188 个，
    # 其余 170 个 reward≈0（谷跑到带外、被 min_edge_headroom 门控清零）。
    # 用这些样本做行为克隆，等于 90% 的时间在教策略生成错误目标。
    _min_reward_raw = conf.get("pretrain.min_reward", None)
    min_reward = None if _min_reward_raw is None else float(_min_reward_raw)
    if min_reward is not None:
        keep = rewards[ranked] >= min_reward
        n_before = len(ranked)
        ranked = ranked[keep]
        _logger.info(
            f"[pretrain] reward filter min_reward={float(min_reward):.2f}: "
            f"{len(ranked)}/{n_before} of top-{k} designs kept")
        if len(ranked) == 0:
            raise RuntimeError(
                f"[pretrain] min_reward={min_reward} 过滤后没有剩余样本，"
                f"请降低该阈值或检查 reward 配置")

    # 可选：在 top reward 集合里再做一次多样性筛选。训练集里近重复设计很多
    # （top 5% 的最近邻汉明距离中位数仅 ~16/855 bit），直接做行为克隆会让
    # 策略塌缩到少数几个模板。min_hamming 为 0 时关闭。
    min_hamming = conf.get_int("pretrain.min_hamming", 0)
    if min_hamming > 0:
        flat = topo[ranked].reshape(len(ranked), -1)
        selected = [0]
        for i in range(1, len(ranked)):
            if (flat[i] != flat[selected]).sum(dim=1).min().item() >= min_hamming:
                selected.append(i)
        ranked = ranked[_torch.tensor(selected, device=ranked.device)]
        _logger.info(
            f"[pretrain] diversity filter min_hamming={min_hamming}: "
            f"{len(selected)}/{len(ranked)} designs kept")

    bits = ((topo[ranked] + 1) / 2).round().to(_torch.long).cpu()       # 0/1
    _logger.info(
        f"[pretrain] selected top {k}/{len(ds)} designs by true reward "
        f"(mean={rewards[ranked].mean().item():.2f}, max={rewards[ranked].max().item():.2f})")
    return bits


def prepare_for_training(conf: ConfigTree, output_dir: str, local_rank: int):
    policy_model: nn.Module = MODEL.build_from(conf.get("policy_model"))
    _logger.info(f"Polocy Model Parameters: {compute_nparam(policy_model)/1e6:.2f}M")

    surrogate_model_config = conf.get("surrogate_model")
    load_from = surrogate_model_config.pop("load_from")
    surrogate_model: nn.Module = MODEL.build_from(surrogate_model_config)
    if load_from is not None:
        surrogate_model.load_state_dict(torch.load(load_from, map_location="cpu")["model"])
    _logger.info(f"Surrogate Model Parameters: {compute_nparam(surrogate_model)/1e6:.2f}M")

    optimizer = OPTIMIZER.build_from(conf.get("optimizer"), dict(params=policy_model.parameters()))
    
    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    if torch.cuda.is_available():
        policy_model = policy_model.to(device=get_device())
        surrogate_model = surrogate_model.to(device=get_device())
        _logger.info(f"Move model to device: {get_device()}")

    reward_impl = REWARD.build_from(conf.get("reward"))

    writer = SummaryWriter(output_dir)

    metric_store = MetricsStore(dominant_metric_name="train/policy_loss", max_is_best=False)
    
    states = dict(model=policy_model, optimizer=optimizer, scheduler=scheduler)
    state_ckpt = StateCheckPoint(output_dir)

    state_ckpt.restore(metric_store, states, device=get_device())

    # ---- 策略监督预训练（可选）----
    # 从未加载 checkpoint 时，用训练集 top 设计热启动，避免从均匀随机策略
    # 起步时 GRPO 组内方差过小（report.md 第 3.1 / 11.1 节）。
    pretrain_epochs = conf.get_int("pretrain.epochs", 0)
    if pretrain_epochs and metric_store.total_epoch == 0:
        designs = _load_pretrain_designs(conf, surrogate_model, reward_impl, get_device())
        pretrain_policy(
            policy_model, designs, device=get_device(),
            epochs=pretrain_epochs,
            batch_size=conf.get_int("pretrain.batch_size", 64),
            lr=conf.get_float("pretrain.lr", 1e-3),
            log_interval=conf.get_int("pretrain.log_interval", 10),
        )

    return policy_model, surrogate_model, optimizer, scheduler, reward_impl, \
        writer, metric_store, state_ckpt, states


def _init(local_rank: int, ngpus_per_node: int, args: Args):
    set_proper_device(local_rank)
    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")


    if set_reproducible:
        set_reproducible(generate_random_seed())
    else:
        set_cudnn_auto_tune()
        disable_debug_api()

    create_code_snapshot(name="code", include_suffix=[".py", ".conf"],
                         source_directory=".", store_directory=args.output_dir)

    _logger.info("Collect envs from system:\n" + get_pretty_env_info())
    _logger.info("Args:\n" + pprint.pformat(dataclasses.asdict(args)))



def main_worker(local_rank: int,
                ngpus_per_node: int,
                args: Args,
                conf: ConfigTree):

    _init(local_rank=local_rank, ngpus_per_node=ngpus_per_node, args=args)

    policy_model, surrogate_model, optimizer, scheduler, reward_impl, \
        writer, metric_store, state_ckpt, states = \
        prepare_for_training(conf, args.output_dir, local_rank)

    run_one_epoch_grpo(
        policy_model=policy_model,
        surrogate_model=surrogate_model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=get_device(),
        log_interval=conf.get_int("log_interval"),
        save_interval=conf.get_int("save_interval"),
        # RL configs
        max_episodes=conf.get_int("max_episodes"),
        batch_size=conf.get_int("rl.batch_size"),
        reward_fn=reward_impl,
        entropy_coef=conf.get_float("rl.entropy_coef", 0.0),
        adv_eps=conf.get_float("rl.adv_eps", 1e-8),
        grad_clip_norm=conf.get_float("rl.grad_clip_norm", None),
        #
        writer=writer,
        metric_store=metric_store,
        state_ckpt=state_ckpt,
        states=states,
    )
    
    # 采样一次，同时用于 markdown 与 TensorBoard 导出
    top_samples = sample_topk_policy_samples(
        policy_model=policy_model,
        surrogate_model=surrogate_model,
        reward_fn=reward_impl,
        device=get_device(),
        n_samples = conf.get_int("infer.n_samples"),
        batch_size = conf.get_int("infer.batch_size"),
        top_k = conf.get_int("infer.top_k"),
    )

    export_topk_policy_samples_to_markdown(
        policy_model=policy_model,
        surrogate_model=surrogate_model,
        reward_fn=reward_impl,
        device=get_device(),
        out_md_path=args.output_dir / "topk_samples.md",
        # sampling config
        n_samples = conf.get_int("infer.n_samples"),
        batch_size = conf.get_int("infer.batch_size"),
        top_k = conf.get_int("infer.top_k"),
        precomputed_samples=top_samples,
    )

    # 把 Top-K 的 S 参数曲线写入 TensorBoard（每个设计一个 run）
    export_topk_policy_samples_to_tensorboard(
        top_samples=top_samples,
        log_dir=args.output_dir,
        reward_fn=reward_impl,
        tag_prefix="topk_design",
    )
    writer.flush()

    

def main(args: Args):
    ngpus_per_node = torch.cuda.device_count()
    local_rank = 0
    main_worker(local_rank, ngpus_per_node, args, args.conf)

if __name__ == "__main__":
    main(get_args())