import torch
from typing import Sequence, Optional, Tuple
from src.registry import REWARD


@REWARD.register
class MultiSParamBandReward:
    """
    指定频段内、三个 S 参数幅值(dB)的加权 reward（面向 GRPO 训练）。

    约定
    ----
    - 输入是代理模型输出的 S 参数幅度，单位为 dB，形状 (B, 3 * num_points)
      或已经是 (B, 3, num_points)；三条曲线的顺序由 s_param_order 指定
    - 频率轴与训练数据一致：freq_start_ghz ~ freq_end_ghz 共 num_points 点（均匀）
    - 通道顺序：[S11, S21, S22]

    Reward
    ------
    只统计 bands 指定的频段（默认 6.4 ~ 7.1 GHz），对频段内取平均：

        reward = -w11 * mean(S11_dB) - w22 * mean(S22_dB) + w21 * mean(S21_dB)

    因为 dB 是负值：
    - S11 / S22 越负（越小）=> -mean 越大 => reward 越高，符合“越小越好”
    - S21 越接近 0（越大）  => +mean 越大 => reward 越高，符合“越大越好”

    频段索引用 round 计算（而不是浮点比较），避免 linspace 的
    7.1000000000000005 这类误差把边界点漏掉。
    """

    def __init__(
        self,
        bands: Sequence[float],              # [f_start, f_end]，单位 GHz
        freq_start_ghz: float,
        freq_end_ghz: float,
        num_points: int,
        s_param_order: Sequence[str] = ("S11", "S21", "S22"),
        weight_s11: float = 1.0,
        weight_s22: float = 1.0,
        weight_s21: float = 1.0,
        clamp_db: Optional[Sequence[float]] = (-60.0, 0.0),
    ):
        if len(bands) != 2:
            raise ValueError("bands 必须是长度为 2 的序列，例如 [6.4, 7.1]")

        f_start, f_end = float(bands[0]), float(bands[1])
        if f_end < f_start:
            raise ValueError(f"频段范围错误：[{f_start}, {f_end}]")

        order = tuple(str(s).upper() for s in s_param_order)
        if sorted(order) != ["S11", "S21", "S22"]:
            raise ValueError(
                f"s_param_order 必须包含 S11/S21/S22 各一次，当前为 {order}")

        self.band = (f_start, f_end)
        self.freq_start = float(freq_start_ghz)
        self.freq_end = float(freq_end_ghz)
        self.num_points = int(num_points)
        if self.num_points < 2:
            raise ValueError(f"num_points 至少为 2，当前为 {self.num_points}")
        if not self.freq_start < self.freq_end:
            raise ValueError(
                f"频率范围错误：[{self.freq_start}, {self.freq_end}]")

        self.s_param_order = order
        self.weight_s11 = float(weight_s11)
        self.weight_s22 = float(weight_s22)
        self.weight_s21 = float(weight_s21)
        self.clamp_db: Optional[Tuple[float, float]] = (
            (float(clamp_db[0]), float(clamp_db[1])) if clamp_db is not None else None)

        # 频率轴（默认 CPU，可用 .to(device) 迁移）
        self.freqs = torch.linspace(self.freq_start, self.freq_end, self.num_points)

        # 频段索引：用 round 定位，避开浮点边界误差
        self.band_idx = self._band_indices(*self.band)

        # 三条曲线的通道位置，例如 (0, 1, 2) 表示 [S11, S21, S22]
        self._ch = {name: order.index(name) for name in ("S11", "S21", "S22")}

    def _band_indices(self, f_start: float, f_end: float) -> torch.Tensor:
        step = (self.freq_end - self.freq_start) / (self.num_points - 1)
        lo = int(round((f_start - self.freq_start) / step))
        hi = int(round((f_end - self.freq_start) / step))
        if lo < 0 or hi > self.num_points - 1 or hi < lo:
            raise ValueError(
                f"频段 [{f_start}, {f_end}] GHz 不在频率轴 "
                f"[{self.freq_start}, {self.freq_end}] GHz（{self.num_points} 点）内"
            )
        return torch.arange(lo, hi + 1, dtype=torch.long)

    def to(self, device):
        """把内部频率轴与索引迁移到指定 device，避免跨设备索引错误。"""
        self.freqs = self.freqs.to(device)
        self.band_idx = self.band_idx.to(device)
        return self

    def __call__(self, s_curve: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - s_curve: (B, 3 * num_points) 或 (B, 3, num_points)

        输出：
        - reward: (B,)
        """
        if s_curve.ndim == 2:
            expected = 3 * self.num_points
            if s_curve.shape[1] != expected:
                raise ValueError(
                    f"s_curve 形状应为 (B, {expected}) 或 (B, 3, {self.num_points})，"
                    f"当前为 {tuple(s_curve.shape)}"
                )
            s = s_curve.reshape(s_curve.shape[0], 3, self.num_points)
        elif s_curve.ndim == 3:
            if s_curve.shape[1] != 3 or s_curve.shape[2] != self.num_points:
                raise ValueError(
                    f"s_curve 形状应为 (B, 3, {self.num_points})，"
                    f"当前为 {tuple(s_curve.shape)}"
                )
            s = s_curve
        else:
            raise ValueError(
                f"s_curve 应为 2 维或 3 维张量，当前为 {s_curve.ndim} 维")

        if self.clamp_db is not None:
            s = torch.clamp(s, self.clamp_db[0], self.clamp_db[1])

        # 取频段内曲线：(B, 3, N_band)
        band = s[:, :, self.band_idx]

        mean_db = band.mean(dim=2)                       # (B, 3)
        s11 = mean_db[:, self._ch["S11"]]
        s21 = mean_db[:, self._ch["S21"]]
        s22 = mean_db[:, self._ch["S22"]]

        reward = (-self.weight_s11 * s11
                  - self.weight_s22 * s22
                  + self.weight_s21 * s21)
        return reward
