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

from src.config import Args, get_args
from src.registry import DATA, MODEL, OPTIMIZER, SCHEDULER, LOSS
from src.surrogate_model_training.engine import train_one_epoch, evaluate_one_epoch, ModelEMA

from src.torchutils.common import set_cudnn_auto_tune, set_reproducible, generate_random_seed, disable_debug_api
from src.torchutils.common import set_proper_device, get_device
from src.torchutils.common import compute_nparam
from src.torchutils.common import StateCheckPoint
from src.torchutils.common import MetricsStore
from src.torchutils.metrics import EstimatedTimeArrival
from src.torchutils.logging import init_logger, create_code_snapshot


_logger = logging.getLogger(__name__)

# 代理模型输出的三条曲线顺序，与 data/build_dataset.py 的 s_param_order 一致
S_PARAM_NAMES = ("s11", "s21", "s22")


def _write_point_wise_l1(writer: SummaryWriter, metric, val_loader, epoch: int) -> None:
    """
    把逐频点的 L1 误差按 S11 / S21 / S22 拆成三条曲线写入 TensorBoard，
    便于看出哪一段频带拟合得不好。

    两个关键点：
      1) add_scalar 的 step 必须是整数。旧写法直接传 float 频率
         (7.5, 7.6, ...)，会被截断成 7,7,...,8,8，101 个频点被挤成
         11 个台阶，曲线看起来是阶梯状。这里改用 MHz 作 step
         (7.5 GHz -> 7500)，既保留分辨率又是整数。
         横轴单位为 MHz，7500 表示 7.5 GHz。
      2) 该曲线只在训练结束后写一次（写最优 epoch 的那份），否则每个
         epoch 都往同一批 step 追加一个点，同一条竖线上会堆上 180 个值。

    metric 是某次 eval 中每个 batch 的 (3 * N,) 张量组成的列表。
    """
    if len(metric) == 0:
        return

    point_wise = torch.stack(metric).mean(dim=0).flatten()
    n_s = len(S_PARAM_NAMES)
    if point_wise.numel() % n_s != 0:
        for i, value in enumerate(point_wise.tolist()):
            writer.add_scalar("eval/l1_distance/all", value, i)
        return

    n_freq = point_wise.numel() // n_s
    curves = point_wise.reshape(n_s, n_freq)
    freq = getattr(getattr(val_loader, "dataset", None), "freq_ghz", None)

    for si, sname in enumerate(S_PARAM_NAMES):
        for fi in range(n_freq):
            if freq is not None and fi < len(freq):
                # MHz -> 整数，避免 float 被截断成台阶
                step = int(round(float(freq[fi]) * 1000))
            else:
                step = fi
            writer.add_scalar(f"eval/l1_distance/{sname}", curves[si, fi].item(), step)


def excute_pipeline(
    only_evaluate: bool,
    start_epoch: int,
    max_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    writer: SummaryWriter,
    metric_store: MetricsStore,
    state_ckpt: StateCheckPoint,
    states: dict,
    early_stop_patience: int = None,
    early_stop_min_delta: float = 0.0,
    **kwargs
):
    if only_evaluate:
        metric_store += evaluate_one_epoch(
            epoch=0,
            loader=val_loader,
            **kwargs
        )
        return

    eta = EstimatedTimeArrival(max_epochs)

    # 早停：连续 early_stop_patience 个 epoch 验证损失没有改善就停，
    # 避免在已经过拟合的区间里空转（旧版 90 epoch 中后 31 个 epoch 毫无改善）。
    epochs_without_improvement = 0
    best_val_so_far = float("inf")
    # 逐频点误差只在训练结束后写一次，避免同一批 step 被重复叠加
    best_point_wise = None

    for epoch in range(start_epoch+1, max_epochs+1):

        metric_store += train_one_epoch(
            epoch=epoch,
            loader=train_loader,
            **kwargs
        )

        metric_store += evaluate_one_epoch(
            epoch=epoch,
            loader=val_loader,
            **kwargs
        )

        for name, metric in metric_store.get_last_metrics().items():
            if "point_wise_l1_distance_list" not in name:
                writer.add_scalar(name, metric, epoch)

        # 保存时用 EMA 权重（若启用）作为 model 的状态，这样 best_checkpoint.pt
        # 里就是真正用于推理/当奖励函数的平均权重，而不是最后一轮的原始权重。
        ema = kwargs.get("ema")
        if ema is not None:
            states["model"].load_state_dict(ema.state_dict())
        state_ckpt.save(metric_store=metric_store, states=states)

        eta.step()

        best_metrics = metric_store.get_best_metrics()
        val_loss = metric_store.get_last_metrics()["eval/loss"]
        if val_loss < best_val_so_far - early_stop_min_delta:
            best_val_so_far = val_loss
            epochs_without_improvement = 0
            best_point_wise = metric_store.get_last_metrics().get(
                "eval/point_wise_l1_distance_list")
        else:
            epochs_without_improvement += 1

        _logger.info(f"Epoch={epoch:04d} complete, val loss={val_loss:.8f}, "
                     f"best val loss={best_metrics['eval/loss']:.8f} "
                     f"(epoch={metric_store.best_epoch+1}), "
                     f"no-improve={epochs_without_improvement}, {eta}")

        if early_stop_patience is not None and epochs_without_improvement >= early_stop_patience:
            _logger.info(
                f"Early stopping at epoch={epoch:04d}: val loss has not improved for "
                f"{epochs_without_improvement} epochs (best={best_val_so_far:.8f} "
                f"at epoch={metric_store.best_epoch+1}).")
            break

    if best_point_wise is not None:
        _write_point_wise_l1(writer, best_point_wise, val_loader,
                             metric_store.best_epoch + 1)
        writer.flush()


def prepare_for_training(conf: ConfigTree, output_dir: str, local_rank: int):
    model: nn.Module = MODEL.build_from(conf.get("model"))
    _logger.info(f"Model Parameters: {compute_nparam(model)/1e6:.2f}M")

    train_loader, val_loader = DATA.build_from(conf.get("data"))

    criterion = LOSS.build_from(conf.get("loss"))
    _logger.info(f"Loss: {type(criterion).__name__}"
                 + (f" (lambda_passivity={criterion.lambda_passivity})"
                    if hasattr(criterion, "lambda_passivity") else ""))

    optimizer_config: dict = conf.get("optimizer")
    
    optimizer = OPTIMIZER.build_from(optimizer_config, dict(params=model.parameters()))
    
    scheduler = SCHEDULER.build_from(conf.get("scheduler"), dict(optimizer=optimizer))

    if torch.cuda.is_available():
        model = model.to(device=get_device())
        criterion = criterion.to(device=get_device())

    writer = SummaryWriter(output_dir)

    metric_store = MetricsStore(dominant_metric_name="eval/loss", max_is_best=False)
    
    states = dict(model=model, optimizer=optimizer, scheduler=scheduler)
    state_ckpt = StateCheckPoint(output_dir)

    state_ckpt.restore(metric_store, states, device=get_device())

    return model, train_loader, val_loader, criterion, optimizer, scheduler, \
        writer, metric_store, state_ckpt, states


def _init(local_rank: int, ngpus_per_node: int, args: Args):
    set_proper_device(local_rank)
    rank = args.node_rank*ngpus_per_node+local_rank
    init_logger(rank=rank, filenmae=args.output_dir/"default.log")


    # 注意：这里判断的必须是配置项，而不是导入的同名函数对象
    # （旧写法 `if set_reproducible:` 判断的是函数本身，恒为真，导致该配置被忽略）
    if args.conf.get_bool("set_reproducible"):
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

    model, train_loader, val_loader, criterion, optimizer, scheduler, \
        writer, metric_store, state_ckpt, states = \
        prepare_for_training(conf, args.output_dir, local_rank)

    ema_decay = conf.get_float("ema_decay", None)
    ema = ModelEMA(model, decay=ema_decay) if ema_decay is not None else None
    if ema is not None:
        _logger.info(f"Enable EMA of model weights, decay={ema_decay}")

    excute_pipeline(
        only_evaluate=conf.get_bool("only_evaluate"),
        start_epoch=metric_store.total_epoch,
        max_epochs=conf.get_int("max_epochs"),
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        metric_store=metric_store,
        state_ckpt=state_ckpt,
        states=states,
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=get_device(),
        log_interval=conf.get_int("log_interval"),
        grad_clip_norm=conf.get_float("grad_clip_norm", None),
        ema=ema,
        early_stop_patience=conf.get_int("early_stop_patience", None),
        early_stop_min_delta=conf.get_float("early_stop_min_delta", 0.0),
    )


def main(args: Args):
    ngpus_per_node = torch.cuda.device_count()
    local_rank = 0
    main_worker(local_rank, ngpus_per_node, args, args.conf)

if __name__ == "__main__":
    main(get_args())