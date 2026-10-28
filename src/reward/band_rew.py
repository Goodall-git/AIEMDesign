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
        self.min_edge_headroom_db = float(min_edge_headroom_db)
        self.weight_min_position = float(weight_min_position)

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


@REWARD.register
class MatchingNetworkReward:
    """
    匹配网络（matching network）reward：**带内深极小值** + 低插损。

    与 BandpassFilterReward 的本质区别
    ----------------------------------
    带通滤波器看的是「带内通、带外阻」；匹配网络看的是「在目标频段内
    把 S11 / S22 匹配到很深的极小值」。旧 reward 用带内 **平均** S11/S22，
    而真实数据里 S11 的带内最小值与平均值相关系数高达 1.00 —— 说明
    「平均」这一指标既不能区分「整段均匀匹配」和「某一点谐振匹配」，
    也不能表达「极小值必须落在带内」这一要求。

    本类直接把「带内最小值」作为匹配项，并额外奖励「极值点位于带内
    而不是带边」：

        reward = + w_s21_pass * mean(S21_dB[带内])
                 - w_s11_min  * cap_min(S11_dB[带内])
                 - w_s22_min  * cap_min(S22_dB[带内])
                 + w_dip      * dip_bonus(S11) + w_dip * dip_bonus(S22)
                 - w_s21_stop * agg(S21_dB[带外])          # 可选

    dip_bonus 定义（每条曲线）：

        edge = min(S11[带内第一个点], S11[带内最后一个点])   # 取较浅的那个边
        inner = min(S11[带内中间点])
        dip_bonus = relu(edge - inner - dip_margin)

    即「两端都明显高于带内最深处」才给分，直接编码「极小值在带内」。
    取 min(edges) 可排除「从带边一路单调下降」的伪凹陷。

    为什么用 cap_min
    ----------------
    dB 值可以一直降到 -60（数据截断下界），如果直接对最小值线性加权，
    RL 会去找「整段压到 -60 dB」的退化解（report.md 反复出现的 reward
    hacking）。cap_min 把超过 min_target_db 的部分截断：

        cap_min(x) = max(x, min_target_db)

    例如 min_target_db=-25 时，-30 dB 与 -60 dB 得分相同，策略没有动机
    继续往 -60 冲；同时 -20 dB 仍比 -25 dB 差，保留梯度。

    参数
    ----
    passband : [f_lo, f_hi]，单位 GHz
    min_target_db : 匹配深度上限（默认 -25 dB），达到后不再加分
    w_dip : 「极值在带内」的奖励权重（默认 0.5，设 0 关闭）
    w_s21_stop : 带外抑制权重（默认 0；匹配网络通常不需要）
    weight_out_of_band_penalty : 带外 S11/S22 比带内更深时的惩罚权重
                                 （默认 0 关闭）。用于禁止「带外主谐振」：
                                 实测 7 GHz 任务在只约束带内时，策略会生成
                                 带内 -24 dB / 带外 8.9 GHz -29 dB 的版图，
                                 即真正的谐振点跑到带外，带内只是个副瓣。
    out_of_band_margin_db : 允许带外比带内深多少 dB 才不罚（默认 0）
    """

    def __init__(
        self,
        passband: Sequence[float],
        freq_start_ghz: float,
        freq_end_ghz: float,
        num_points: int,
        s_param_order: Sequence[str] = ("S11", "S21", "S22"),
        weight_s21_pass: float = 1.0,
        weight_s11_min: float = 1.0,
        weight_s22_min: float = 1.0,
        min_target_db: float = -25.0,
        weight_dip: float = 0.5,
        dip_margin_db: float = 0.0,
        weight_s21_stop: float = 0.0,
        stopband_aggregation: str = "mean",
        # ---- 凹陷形状约束 ----
        # 只用 min() 会奖励「从带边一路单调下降」的伪凹陷：RL 实测找到的
        # top-k 就是 S11 = -23.5 → -26.5 的单调曲线，min 达标但带内没有
        # 任何极小值。下面两项把「必须是内部凹陷」写进 reward：
        #
        #   min_edge_headroom_db: 两端都要比内部最深处浅至少这么多 dB
        #   weight_min_position : 最深处越靠近带中心，额外奖励越多
        min_edge_headroom_db: float = 3.0,
        weight_min_position: float = 0.0,
        max_dip_bonus_db: Optional[float] = None,
        # ---- 带外谐振抑制（防止真正的谐振点跑到带外）----
        # 只约束「带内要有深极小值」是不够的：策略可以让带内只做到 -24 dB，
        # 同时在 8.9 GHz 造一个 -29 dB 的更深谐振点来「省事」。
        # 这样的版图在物理上主谐振在带外，带内只是副瓣，完全不能用。
        # 打开后对 S11/S22 的「带外最深处 vs 带内最深处」做 hinge 惩罚。
        weight_out_of_band_penalty: float = 0.0,
        out_of_band_margin_db: float = 0.0,
        clamp_db: Optional[Sequence[float]] = (-60.0, 0.0),
    ):
        if len(passband) != 2:
            raise ValueError("passband 必须是长度为 2 的序列，例如 [6.4, 7.1]")
        f_lo, f_hi = float(passband[0]), float(passband[1])
        if f_hi <= f_lo:
            raise ValueError(f"通带范围错误：[{f_lo}, {f_hi}]")

        order = tuple(str(s).upper() for s in s_param_order)
        if sorted(order) != ["S11", "S21", "S22"]:
            raise ValueError(
                f"s_param_order 必须包含 S11/S21/S22 各一次，当前为 {order}")

        agg = str(stopband_aggregation).lower()
        if agg not in ("mean", "max"):
            raise ValueError(
                f"stopband_aggregation 只能是 mean 或 max，当前为 {stopband_aggregation}")

        self.passband = (f_lo, f_hi)
        self.freq_start = float(freq_start_ghz)
        self.freq_end = float(freq_end_ghz)
        self.num_points = int(num_points)
        if self.num_points < 2:
            raise ValueError(f"num_points 至少为 2，当前为 {self.num_points}")

        self.s_param_order = order
        self.weight_s21_pass = float(weight_s21_pass)
        self.weight_s11_min = float(weight_s11_min)
        self.weight_s22_min = float(weight_s22_min)
        self.min_target_db = float(min_target_db)
        self.weight_dip = float(weight_dip)
        self.dip_margin_db = float(dip_margin_db)
        self.weight_s21_stop = float(weight_s21_stop)
        self.stopband_aggregation = agg
        self.min_edge_headroom_db = float(min_edge_headroom_db)
        self.weight_min_position = float(weight_min_position)
        self.max_dip_bonus_db = None if max_dip_bonus_db is None else float(max_dip_bonus_db)
        self.weight_out_of_band_penalty = float(weight_out_of_band_penalty)
        self.out_of_band_margin_db = float(out_of_band_margin_db)
        self.clamp_db: Optional[Tuple[float, float]] = (
            (float(clamp_db[0]), float(clamp_db[1])) if clamp_db is not None else None)

        self.freqs = torch.linspace(self.freq_start, self.freq_end, self.num_points)
        self.band_idx = _band_indices(self.freq_start, self.freq_end,
                                      self.num_points, f_lo, f_hi)
        in_band_mask = torch.zeros(self.num_points, dtype=torch.bool)
        in_band_mask[self.band_idx] = True
        self.stop_idx = torch.arange(self.num_points, dtype=torch.long)[~in_band_mask]

        self._ch = {name: order.index(name) for name in ("S11", "S21", "S22")}

    def to(self, device):
        self.freqs = self.freqs.to(device)
        self.band_idx = self.band_idx.to(device)
        self.stop_idx = self.stop_idx.to(device)
        return self

    def _cap_min(self, x: torch.Tensor) -> torch.Tensor:
        """db 值越小越好；低于 min_target_db 的部分不再加分。"""
        return torch.maximum(x, torch.full_like(x, self.min_target_db))

    def _interior_depth(self, band_curve: torch.Tensor) -> torch.Tensor:
        """
        内部凹陷深度：min(两端) - min(内部点)。

        两端都必须比内部最深处浅，否则（例如单调下降）返回 0。
        这是判断「极小值真的在带内」而不是「最小值恰好落在带边」的关键。
        """
        if band_curve.shape[1] < 3:
            return torch.zeros(band_curve.shape[0], device=band_curve.device,
                               dtype=band_curve.dtype)
        edge = torch.minimum(band_curve[:, 0], band_curve[:, -1])
        inner = band_curve[:, 1:-1].min(dim=1).values
        return torch.relu(edge - inner)

    def _center_score(self, band_curve: torch.Tensor) -> torch.Tensor:
        """最深处距带中心的归一化接近度：中心=1，带边=0。"""
        n = band_curve.shape[1]
        if n < 3:
            return torch.zeros(band_curve.shape[0], device=band_curve.device,
                               dtype=band_curve.dtype)
        pos = band_curve.argmin(dim=1).to(band_curve.dtype)
        center = (n - 1) / 2.0
        return 1.0 - (pos - center).abs() / center

    def _dip_bonus(self, band_curve: torch.Tensor) -> torch.Tensor:
        """
        band_curve: (B, N_band) 的 dB 曲线
        返回 (B,)：**带内真凹陷**的深度（relu 后非负）。

        必须两端都比内部最深处高（即极小值真的落在带内），才给分：

            depth = min(edge_left, edge_right) - min(interior)

        用 min 而不是 max 是关键：若极小值落在带边（例如从 6.4 GHz 一路
        下降到 7.1 GHz），一端本身就很深，min(edges) 会把它排除，不会误判为
        「带内凹陷」。旧写法用 max 会把这种单调下降也当成凹陷并给高分。

        N_band 太小时（<3）无法定义，返回 0。
        """
        if band_curve.shape[1] < 3:
            return torch.zeros(band_curve.shape[0], device=band_curve.device,
                               dtype=band_curve.dtype)
        edge = torch.minimum(band_curve[:, 0], band_curve[:, -1])
        inner = band_curve[:, 1:-1].min(dim=1).values
        bonus = torch.relu(edge - inner - self.dip_margin_db)
        if self.max_dip_bonus_db is not None:
            bonus = torch.clamp(bonus, max=self.max_dip_bonus_db)
        return bonus

    def __call__(self, s_curve: torch.Tensor) -> torch.Tensor:
        s = _as_three_curves(s_curve, self.num_points)
        if self.clamp_db is not None:
            s = torch.clamp(s, self.clamp_db[0], self.clamp_db[1])

        # ---------- 带内 ----------
        s11_band = s[:, self._ch["S11"], :][:, self.band_idx]   # (B, N_band)
        s21_band = s[:, self._ch["S21"], :][:, self.band_idx]
        s22_band = s[:, self._ch["S22"], :][:, self.band_idx]

        s11_min = s11_band.min(dim=1).values
        s22_min = s22_band.min(dim=1).values
        s21_mean = s21_band.mean(dim=1)

        # 「带内极小值」门控：只有确实存在内部凹陷时，匹配项才全额计分。
        # 单调下降（min 落在带边）不算带内极小值，避免旧 reward 那种
        # 「min 达标但曲线一路下滑」的伪解。
        gate11 = (self._interior_depth(s11_band) >= self.min_edge_headroom_db).to(s11_min.dtype)
        gate22 = (self._interior_depth(s22_band) >= self.min_edge_headroom_db).to(s22_min.dtype)

        reward = (self.weight_s21_pass * s21_mean
                  - self.weight_s11_min * self._cap_min(s11_min) * gate11
                  - self.weight_s22_min * self._cap_min(s22_min) * gate22)

        if self.weight_dip != 0.0:
            reward = reward + self.weight_dip * (
                self._dip_bonus(s11_band) + self._dip_bonus(s22_band))

        if self.weight_min_position != 0.0:
            reward = reward + self.weight_min_position * (
                self._center_score(s11_band) + self._center_score(s22_band))

        # ---------- 带外 S21 抑制（可选，用于滤波器）----------
        if self.weight_s21_stop != 0.0 and self.stop_idx.numel() > 0:
            s21_out = s[:, self._ch["S21"], :][:, self.stop_idx]
            agg = (s21_out.max(dim=1).values if self.stopband_aggregation == "max"
                   else s21_out.mean(dim=1))
            reward = reward - self.weight_s21_stop * agg

        # ---------- 带外谐振抑制（S11 / S22）----------
        # 目的：禁止「带外有更深的主谐振点」。
        #
        # 之前的 reward 只奖励「带内最小值够深」，没有管带外，于是策略
        # 找到了一个省事的解：带内做到 -24 dB 就行（反正 -25 dB 就封顶了），
        # 同时在 8.9 GHz 造一个 -29 dB 的更深谐振。这种版图的主谐振其实
        # 在带外，带内的「谷」只是次级副瓣，实物上不能用。
        #
        # 这里对每个通道算：
        #   excess = relu(带外最深 - 带内最深 - margin)
        # 带外不比带内更深时 excess=0，不罚；一旦更深就按超出量线性惩罚。
        # 用 relu 而不是常数惩罚，保证梯度方向明确（把带外那个谷抬起来）。
        if self.weight_out_of_band_penalty != 0.0 and self.stop_idx.numel() > 0:
            for name, gate in (("S11", gate11), ("S22", gate22)):
                in_min = s[:, self._ch[name], :][:, self.band_idx].min(dim=1).values
                out_min = s[:, self._ch[name], :][:, self.stop_idx].min(dim=1).values
                # 只有「带内已经做出合格凹陷」时才追究带外。
                # 否则策略会走退化解：把整条曲线压平（带内不带外都不深），
                # 惩罚恒为 0 却不满足任何匹配要求 —— 实测 w=1 时 reward
                # 直接从 57 掉到 -0.44，S11 带内只剩 -2.5 dB。
                # 乘 gate 后，惩罚只作用在「已经拿到带内分」的个体上，
                # 等价于"想要带内的高分，就别在带外造更深的主谐振"。
                excess = torch.relu(in_min - out_min - self.out_of_band_margin_db)
                reward = reward - self.weight_out_of_band_penalty * excess * gate

        return reward
