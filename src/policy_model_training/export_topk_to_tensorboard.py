"""
把已训练好的策略模型采样出的 Top-K 版图，其**完整 S 参数曲线**写入指定 TensorBoard
日志目录（每个设计一个 run），便于在 TensorBoard 里直接对比前 K 名设计的 S11/S21/S22。

典型用法（对已有的 RL 训练输出补写）：

    python -m src.policy_model_training.export_topk_to_tensorboard \
        --conf conf/rl.conf \
        --rl-output rl_output/01 \
        --policy-ckpt rl_output/01/checkpoint.pt \
        --top-k 5

写入后：

    tensorboard --logdir=rl_output/

在 S11 / S21 / S22 三张图中会各出现 K 条曲线，run 名形如
`topk_design_rank01` ~ `topk_design_rank0K`（rank01 = reward 最高）。

注意：这里写的是**代理模型的预测曲线**（与 topk_samples.md 同源），
其可信度取决于代理模型在对应输入分布上的外推能力，请结合 report.md 判读。
"""

import argparse
import logging
import pathlib

import torch
import torch.nn as nn
from pyhocon import ConfigFactory

import src.models
import src.reward
from src.registry import MODEL, REWARD
from src.policy_model_training.infer import (
    sample_topk_policy_samples,
    export_topk_policy_samples_to_tensorboard,
)
from src.torchutils.common import set_proper_device, get_device


_logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conf", default="conf/rl.conf", help="RL 配置文件")
    parser.add_argument("--rl-output", default="rl_output/01",
                        help="TensorBoard 日志目录（即 RL 训练输出目录）")
    parser.add_argument("--policy-ckpt", default=None,
                        help="策略模型 checkpoint，默认 <rl-output>/checkpoint.pt")
    parser.add_argument("--surrogate-ckpt", default=None,
                        help="代理模型 checkpoint，默认取 conf 里的 surrogate_model.load_from")
    parser.add_argument("--top-k", type=int, default=5, help="写入 TensorBoard 的设计数量")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="采样数量，默认取 conf 的 infer.n_samples")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="采样 batch，默认取 conf 的 infer.batch_size")
    parser.add_argument("--tag-prefix", default="topk_design", help="TensorBoard tag 前缀")
    parser.add_argument("--device", default=None, help="cuda 序号或 cpu，默认自动选择")
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s : %(message)s")
    args = parse_args()

    conf = ConfigFactory.parse_file(args.conf)
    rl_output = pathlib.Path(args.rl_output)
    policy_ckpt = pathlib.Path(args.policy_ckpt) if args.policy_ckpt \
        else rl_output / "checkpoint.pt"

    # ---------- device ----------
    if args.device is None:
        if torch.cuda.is_available():
            set_proper_device(0)
        else:
            set_proper_device(0)  # 无 cuda 时会落到 "cpu"
    elif args.device != "cpu":
        set_proper_device(int(args.device))
    device = args.device if args.device is not None else get_device()
    _logger.info(f"Using device: {device}")

    # ---------- policy model ----------
    policy_model: nn.Module = MODEL.build_from(conf.get("policy_model"))
    if not policy_ckpt.exists():
        raise FileNotFoundError(f"找不到策略模型 checkpoint: {policy_ckpt}")
    policy_state = torch.load(policy_ckpt, map_location="cpu")
    policy_model.load_state_dict(policy_state["model"])
    _logger.info(f"Loaded policy model from {policy_ckpt}")
    policy_model.to(device)

    # ---------- surrogate model ----------
    surrogate_cfg = conf.get("surrogate_model")
    load_from = surrogate_cfg.pop("load_from")
    surrogate_ckpt = args.surrogate_ckpt or load_from
    surrogate_model: nn.Module = MODEL.build_from(surrogate_cfg)
    surrogate_model.load_state_dict(torch.load(surrogate_ckpt, map_location="cpu")["model"])
    _logger.info(f"Loaded surrogate model from {surrogate_ckpt}")
    surrogate_model.to(device)

    # ---------- reward ----------
    reward_fn = REWARD.build_from(conf.get("reward"))
    if hasattr(reward_fn, "to"):
        reward_fn.to(device)

    # ---------- sampling config ----------
    n_samples = args.n_samples if args.n_samples is not None \
        else conf.get_int("infer.n_samples")
    batch_size = args.batch_size if args.batch_size is not None \
        else conf.get_int("infer.batch_size")

    # ---------- sample top-k ----------
    top_samples = sample_topk_policy_samples(
        policy_model=policy_model,
        surrogate_model=surrogate_model,
        reward_fn=reward_fn,
        device=device,
        n_samples=n_samples,
        batch_size=batch_size,
        top_k=args.top_k,
    )
    _logger.info(f"Sampled {n_samples} designs, selected top-{len(top_samples)}: "
                 + ", ".join(f"#{i+1}={s.reward:.4f}" for i, s in enumerate(top_samples)))

    # ---------- write to tensorboard ----------
    export_topk_policy_samples_to_tensorboard(
        top_samples=top_samples,
        log_dir=rl_output,
        reward_fn=reward_fn,
        tag_prefix=args.tag_prefix,
    )
    _logger.info(f"Done. TensorBoard logdir: {rl_output.resolve()}")


if __name__ == "__main__":
    main()
