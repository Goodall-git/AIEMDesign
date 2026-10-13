import logging
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
from src.policy_model_training.infer import export_topk_policy_samples_to_markdown

from src.torchutils.common import set_cudnn_auto_tune, set_reproducible, generate_random_seed, disable_debug_api
from src.torchutils.common import set_proper_device, get_device
from src.torchutils.common import compute_nparam
from src.torchutils.common import StateCheckPoint
from src.torchutils.common import MetricsStore
from src.torchutils.metrics import EstimatedTimeArrival
from src.torchutils.logging import init_logger, create_code_snapshot


_logger = logging.getLogger(__name__)


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
    
    export_topk_policy_samples_to_markdown(
        policy_model=policy_model,
        surrogate_model=surrogate_model,
        reward_fn=reward_impl,
        device=get_device(),
        out_md_path=args.output_dir / "topk_samples.md",
        # sampling config
        n_samples = conf.get_int("infer.n_samples"),
        batch_size = conf.get_int("infer.batch_size"),
        top_k = conf.get_int("infer.top_k")
    )
    

def main(args: Args):
    ngpus_per_node = torch.cuda.device_count()
    local_rank = 0
    main_worker(local_rank, ngpus_per_node, args, args.conf)

if __name__ == "__main__":
    main(get_args())