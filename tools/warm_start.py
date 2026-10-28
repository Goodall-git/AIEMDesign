"""从已有 RL run 的策略权重热启动到一个新的 run 目录。

用法：
    python tools/warm_start.py \
        --src rl_output/19_match_7g_warm/checkpoint.pt \
        --dst rl_output/20_match_7g_15db

做的三件事：
1. 复制 policy / surrogate 权重（只拷 model 段，因为 surrogate 权重不参与训练）
2. 重置 optimizer 状态（旧 run 的 Adam 动量对应的是旧 reward，直接沿用会把
   新 reward 的梯度方向带偏）
3. 重置 scheduler 的 last_epoch / _step_count，让 lr 从配置里的 3e-4 重新开始
   （否则 CosineAnnealing 会带着 last_epoch=10000 直接停在 eta_min）

不复制 metrics，保证 metric_store.total_epoch == 0，从而触发 pretrain 逻辑
（本配置 pretrain.epochs=0，等于不跑行为克隆）。
"""
import argparse
import pathlib

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep-optimizer", action="store_true")
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    src = torch.load(args.src, map_location="cpu", weights_only=False)
    out = {"model": src["model"]}

    if args.keep_optimizer:
        out["optimizer"] = src["optimizer"]
    else:
        # 必须提供 optimizer 段：StateCheckPoint._find_mismatch 是「缺一段就
        # 整份 checkpoint 作废」，只给 model 会被直接忽略。
        # 这里用一个全新的 AdamW 的 state_dict —— state 为空、param_groups 齐全，
        # 等价于「权重沿用、动量归零」。
        fresh = torch.optim.AdamW(
            [v for v in out["model"].values() if isinstance(v, torch.Tensor)]
            or [torch.zeros(1)],
            lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=1e-5)
        out["optimizer"] = fresh.state_dict()

    sched = dict(src["scheduler"])
    sched["last_epoch"] = 0
    sched["_step_count"] = 1
    sched["_is_initial"] = False
    sched["_get_lr_called_within_step"] = False
    out["scheduler"] = sched
    out["metrics"] = {}

    dst = pathlib.Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    torch.save(out, dst / "checkpoint.pt")
    print(f"warm start checkpoint written to {dst/'checkpoint.pt'}  "
          f"(model keys={len(out['model'])}, lr will restart from config)")


if __name__ == "__main__":
    main()
