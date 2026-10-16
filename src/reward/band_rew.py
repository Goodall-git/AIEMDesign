import torch
from typing import Sequence, Optional, Tuple
from src.registry import REWARD


def _as_three_curves(s_curve: torch.Tensor, num_points: int) -> torch.Tensor:
    """把 (B, 3*num_points) 或 (B, 3, num_points) 统一成 (B, 3, num_points)。"""
    if s_curve.ndim == 2:
        expected = 3 * num_points
        if s_curve.shape[1] != expected:
            raise ValueError(
                f"s_curve 形状应为 (B, {expected}) 或 (B, 3, {num_points})，"
                f"当前为 {tuple(s_curve.shape)}")
        return s_curve.reshape(s_curve.shape[0], 3, num_points)
    if s_curve.ndim == 3:
        if s_curve.shape[1] != 3 or s_curve.shape[2] != num_points:
            raise ValueError(
                f"s_curve 形状应为 (B, 3, {num_points})，"
                f"当前为 {tuple(s_curve.shape)}")
        return s_curve
    raise ValueError(f"s_curve 应为 2 维或 3 维张量，当前为 {s_curve.ndim} 维")


def _band_indices(freq_start: float, freq_end: float, num_points: int,
                  f_lo: float, f_hi: float) -> torch.Tensor:
    """
    求频段 [f_lo, f_hi] 在均匀频率轴上的索引。

    用 round 而不是浮点比较：np.linspace(0, 10, 101) 的第 71 个点实际是
    7.1000000000000005，用 `<= 7.1` 判断会把 7.1 GHz 漏掉。
    """
    step = (freq_end - freq_start) / (num_points - 1)
    lo = int(round((f_lo - freq_start) / step))
    hi = int(round((f_hi - freq_start) / step))
    if lo < 0 or hi > num_points - 1 or hi < lo:
        raise ValueError(
            f"频段 [{f_lo}, {f_hi}] GHz 不在频率轴 "
            f"[{freq_start}, {freq_end}] GHz（{num_points} 点）内")
    return torch.arange(lo, hi + 1, dtype=torch.long)


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

    注意：本类**只看带内**，不约束带外。要做滤波器形状（带外要有抑制/滚降）
    请用 BandpassFilterReward。
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
        return _band_indices(self.freq_start, self.freq_end, self.num_points,
                             f_start, f_end)

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
        s = _as_three_curves(s_curve, self.num_points)

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


@REWARD.register
class BandpassFilterReward:
    """
    带通滤波器形状的 reward：**全频带**优化（面向 GRPO 训练）。

    与 MultiSParamBandReward 的区别
    ------------------------------
    旧版只看带内（默认 6.4 ~ 7.1 GHz）三个 S 参数的平均值；要做滤波器就必须
    同时约束带外，否则“带内通、带外也全通”的版图一样能拿高分。
    本类把**整个频率轴（默认 0 ~ 10 GHz，101 点）**都纳入目标。

    优化目标
    --------
    带内（passband，默认 6.4 ~ 7.1 GHz）：

      - S21 越大越好  —— 插损越小越好
      - S11 越小越好  —— 输入匹配越好
      - S22 越小越好  —— 输出匹配越好

    带外（其余全部频点）：

      - S21 越小越好  —— 抑制 / 滚降，这是形成滤波形状的关键

        reward = + w_s21_pass * agg(S21_dB[in])
                 - w_s11_pass * agg(S11_dB[in])
                 - w_s22_pass * agg(S22_dB[in])
                 - w_s21_stop * agg(S21_dB[out])

    带内三项对频点取平均；带外那一项由 stopband_aggregation 决定：
    "mean" 取带外平均（默认），"max" 取带外最差的那一点（更严格，逼所有
    带外频点都压下去，对应“滚降要足够”）。

    关于 dB 与权重
    --------------
    dB 是负值，所以各项符号自然成立：

      - S11/S22 带内越负（越小） => -mean 越大 => reward 越高 ✓
      - S21 带内越接近 0（越大） => +mean 越大 => reward 越高 ✓
      - S21 带外越负（越小）     => -mean 越大 => reward 越高 ✓

    所有项都在同一个 dB 尺度上，因此权重可以直接理解为兑换比，
    例如 w_s21_pass=2、w_s21_stop=1 表示“带内 1 dB 插损 == 带外 2 dB 抑制”。

    ⚠️ 权重失衡会导致 reward hacking
    -------------------------------
    带内有 3 项、带外有 1 项都奖励“越负越好”，只有带内 S21 一项奖励
    “越大越好”。如果让前者总和过大，策略会去找“整条曲线一起压到 -60 dB”
    的版图（例如整块金属屏蔽），而不是真正的滤波器——因为那样能白拿三项分。
    默认权重把带内传输设为 2.0、其余为 1.0，让一个真实滤波器优于这种退化解。

    若需要严格按指标设计（例如“带内插损 < 3 dB、回波 < -10 dB、
    带外抑制 > 20 dB”），把 *_target_db 设上：对应项会变成**只罚不奖**的
    hinge 惩罚，达到指标后不再继续加分，从而彻底消除上面这类刷分动机。
    注意此时若一个 batch 内所有设计都满足指标，奖励会全部为 0，
    GRPO 的组内优势也会全为 0（梯度消失）——这属于“已达标”，可接受。
    """

    def __init__(
        self,
        passband: Sequence[float],            # [f_start, f_end]，单位 GHz
        freq_start_ghz: float,
        freq_end_ghz: float,
        num_points: int,
        s_param_order: Sequence[str] = ("S11", "S21", "S22"),
        # ---- 带内 ----
        weight_s21_pass: float = 2.0,         # 插损：越大越好，权重最高
        weight_s11_pass: float = 1.0,         # 输入匹配：越小越好
        weight_s22_pass: float = 1.0,         # 输出匹配：越小越好
        # ---- 带外 ----
        weight_s21_stop: float = 1.0,         # 抑制/滚降：越小越好
        stopband_aggregation: str = "mean",   # mean | max
        # ---- 可选阈值（None = 关闭，用纯加权；设值后该项变成只罚不奖）----
        target_s21_pass_db: Optional[float] = None,   # 带内插损下限，如 -3.0
        target_s11_pass_db: Optional[float] = None,   # 带内回波上限，如 -10.0
        target_s22_pass_db: Optional[float] = None,   # 同上
        target_s21_stop_db: Optional[float] = None,   # 带外抑制上限，如 -20.0
        clamp_db: Optional[Sequence[float]] = (-60.0, 0.0),
    ):
        if len(passband) != 2:
            raise ValueError("passband 必须是长度为 2 的序列，例如 [6.4, 7.1]")

        f_lo, f_hi = float(passband[0]), float(passband[1])
        if f_hi < f_lo:
            raise ValueError(f"通带范围错误：[{f_lo}, {f_hi}]")

        order = tuple(str(s).upper() for s in s_param_order)
        if sorted(order) != ["S11", "S21", "S22"]:
            raise ValueError(
                f"s_param_order 必须包含 S11/S21/S22 各一次，当前为 {order}")

        agg = str(stopband_aggregation).lower()
        if agg not in ("mean", "max"):
            raise ValueError(
                f"stopband_aggregation 只能是 mean 或 max，当前为 {stopband_aggregation}")

        self.band = (f_lo, f_hi)
        self.passband = (f_lo, f_hi)
        self.freq_start = float(freq_start_ghz)
        self.freq_end = float(freq_end_ghz)
        self.num_points = int(num_points)
        if self.num_points < 2:
            raise ValueError(f"num_points 至少为 2，当前为 {self.num_points}")
        if not self.freq_start < self.freq_end:
            raise ValueError(
                f"频率范围错误：[{self.freq_start}, {self.freq_end}]")

        self.s_param_order = order
        self.weight_s21_pass = float(weight_s21_pass)
        self.weight_s11_pass = float(weight_s11_pass)
        self.weight_s22_pass = float(weight_s22_pass)
        self.weight_s21_stop = float(weight_s21_stop)
        self.stopband_aggregation = agg

        self.target_s21_pass_db = self._opt_float(target_s21_pass_db)
        self.target_s11_pass_db = self._opt_float(target_s11_pass_db)
        self.target_s22_pass_db = self._opt_float(target_s22_pass_db)
        self.target_s21_stop_db = self._opt_float(target_s21_stop_db)

        self.clamp_db: Optional[Tuple[float, float]] = (
            (float(clamp_db[0]), float(clamp_db[1])) if clamp_db is not None else None)

        # 频率轴（默认 CPU，可用 .to(device) 迁移）
        self.freqs = torch.linspace(self.freq_start, self.freq_end, self.num_points)

        # 带内 / 带外索引
        self.band_idx = _band_indices(self.freq_start, self.freq_end,
                                      self.num_points, f_lo, f_hi)
        in_band_mask = torch.zeros(self.num_points, dtype=torch.bool)
        in_band_mask[self.band_idx] = True
        self.stop_idx = torch.arange(self.num_points, dtype=torch.long)[~in_band_mask]

        # 三条曲线的通道位置
        self._ch = {name: order.index(name) for name in ("S11", "S21", "S22")}

    @staticmethod
    def _opt_float(v) -> Optional[float]:
        return None if v is None else float(v)

    def to(self, device):
        """把内部频率轴与索引迁移到指定 device，避免跨设备索引错误。"""
        self.freqs = self.freqs.to(device)
        self.band_idx = self.band_idx.to(device)
        self.stop_idx = self.stop_idx.to(device)
        return self

    def _term(self, value: torch.Tensor, target: Optional[float],
              sign: int) -> torch.Tensor:
        """
        计算单项贡献。

        - target 为 None：纯加权，返回 sign * value（dB 尺度，可正可负）
        - target 不为 None：只罚不奖的 hinge，返回值 <= 0
            sign = +1（越大越好的量，如带内 S21）：低于 target 才罚
            sign = -1（越小越好的量，如带内 S11）：高于 target 才罚
        """
        if target is None:
            return sign * value
        if sign > 0:
            return -torch.relu(target - value)
        return -torch.relu(value - target)

    def _agg(self, x: torch.Tensor) -> torch.Tensor:
        """带外聚合：x 为 (B, N_out) -> (B,)"""
        if self.stopband_aggregation == "max":
            return x.max(dim=1).values
        return x.mean(dim=1)

    def __call__(self, s_curve: torch.Tensor) -> torch.Tensor:
        """
        输入：
        - s_curve: (B, 3 * num_points) 或 (B, 3, num_points)

        输出：
        - reward: (B,)
        """
        s = _as_three_curves(s_curve, self.num_points)

        if self.clamp_db is not None:
            s = torch.clamp(s, self.clamp_db[0], self.clamp_db[1])

        # ---------- 带内 ----------
        band_mean = s[:, :, self.band_idx].mean(dim=2)      # (B, 3)
        s11_in = band_mean[:, self._ch["S11"]]
        s21_in = band_mean[:, self._ch["S21"]]
        s22_in = band_mean[:, self._ch["S22"]]

        reward = (self.weight_s21_pass * self._term(s21_in, self.target_s21_pass_db, +1)
                  + self.weight_s11_pass * self._term(s11_in, self.target_s11_pass_db, -1)
                  + self.weight_s22_pass * self._term(s22_in, self.target_s22_pass_db, -1))

        # ---------- 带外（只看 S21，形成滚降）----------
        if self.stop_idx.numel() > 0:
            s21_out = s[:, self._ch["S21"], :][:, self.stop_idx]   # (B, N_out)
            reward = reward + self.weight_s21_stop * self._term(
                self._agg(s21_out), self.target_s21_stop_db, -1)

        return reward
