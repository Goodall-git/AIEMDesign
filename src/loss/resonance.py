# -*- coding: utf-8 -*-
"""
谐振（带内极小值）加权损失 + 显式凹陷深度监督。

背景
----
在本项目的 4180 条数据里，「带内存在真正极小值」的样本极少：以
8.4~9.3 GHz 为例，S11 只有约 0.5%、S22 约 0.4% 的设计满足「带内局部
极小且比带边深 3 dB」。普通 L1 训练时，这些样本对总损失的贡献不到 1%，
模型会直接忽略它们（实测：带内最小值预测很准，但凹陷深度的相关性很低）。

本损失做两件事：
1. 对含带内谐振的样本加权，让 L1 不再被大量平坦曲线淹没；
2. 直接把「凹陷深度」作为一个回归量监督：
       depth = relu(min(带边两端) - min(带内中间点))
   这与 MatchingNetworkReward 的 _interior_depth 完全一致。只加权 L1
   并不能保证深度被拟合，因为一个 1~2 个频点的窄凹陷对 101 点 L1 的
   贡献很小；显式 depth loss 则直接惩罚「预测凹陷太浅」。

用法（conf）：
    loss {
        type_: ResonanceWeightedLoss
        n_freq: 101
        bands: [[7.5, 9.9], [6.0, 8.0]]   # 多频段（推荐）
        # band: [8.4, 9.3]                # 单频段（旧写法，仍兼容）
        depth_threshold_db: 3.0
        weight: 12.0
        depth_weight: 2.0
        depth_negative_weight: 0.1
        depth_negative_margin_db: 0.5
        depth_clip_db: 12.0
        position_weight: 0.2
        criterion { type_: L1Loss }
    }

为什么要支持多频段
------------------
旧版只接受单个 band。当 band=[7.5, 9.9] 时，7.5 GHz 以下没有任何
「凹陷深度」监督 —— 模型的 L1 在低频段看起来还不错（6~7 GHz 仅 1.28 dB），
但那是因为低频曲线本来就平坦、容易拟合；一旦要求「在 6.5~7.5 GHz 形成
局部极小值」，模型直接输出平线：实测把 9 个真实的 6.5~7.5 GHz 凹陷样本
（真实凹陷 1.0~6.8 dB）喂进去，预测凹陷深度 9 个里 8 个为 0.00 dB，
带内 L1 高达 5.95 dB（整段平均才 0.67 dB）。

原因很简单：6.5~7.5 GHz 有凹陷的样本只占 0.24%，均匀 L1 下它们只能
拿到 0.24% 的梯度。多频段 + 加权后，低频凹陷样本才会真正参与训练。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.registry import LOSS

N_S_PARAMS = 3


@LOSS.register
class ResonanceWeightedLoss(nn.Module):
    """
    加权 L1 + 显式凹陷深度（可选位置）监督。

        loss = weighted_l1
             + depth_weight * mean_channel(depth_loss)
             + position_weight * mean_channel(position_loss)

    depth_loss 对「真凹陷样本」用 SmoothL1 拟合深度，对「非凹陷样本」用
    hinge 抑制虚假凹陷（relu(pred_depth - margin)）。

    真凹陷判据与 MatchingNetworkReward 一致：
      取 band 内全局最小点 p；要求 p 距两端 >= 1 个点；
      depth = min(band[0], band[-1]) - band[p]。
    """

    def __init__(self, criterion, n_freq: int, band=None, bands=None,
                 depth_threshold_db: float = 3.0,
                 edge_margin_points: int = 1, weight: float = 5.0,
                 depth_weight: float = 0.0, depth_negative_weight: float = 0.1,
                 depth_negative_margin_db: float = 0.5, depth_clip_db: float = 12.0,
                 position_weight: float = 0.0, position_temperature: float = 0.5,
                 n_s_params: int = N_S_PARAMS):
        super(ResonanceWeightedLoss, self).__init__()
        if isinstance(criterion, dict) or (hasattr(criterion, "keys") and hasattr(criterion, "get")):
            from src.registry import LOSS as _LOSS
            criterion = _LOSS.build_from(criterion)
        if not isinstance(criterion, nn.Module):
            raise TypeError(f"criterion 需要是 nn.Module 或含 type_ 的配置，实际为 {type(criterion)}")

        self.criterion = criterion
        self.n_freq = int(n_freq)
        self.n_s_params = int(n_s_params)
        # 兼容两种写法：bands=[[a,b],...] 优先；否则用单 band=[a,b]
        if bands is None:
            if band is None:
                raise ValueError("必须提供 bands 或 band 之一")
            bands = [band]
        # 允许 pyhocon 的 ConfigList / 元组 / 单个二元组
        norm = []
        for i, b in enumerate(bands):
            vals = list(b)
            if len(vals) != 2:
                raise ValueError(f"bands[{i}] 必须是长度为 2 的序列，实际为 {vals}")
            lo, hi = float(vals[0]), float(vals[1])
            if hi <= lo:
                raise ValueError(f"bands[{i}] 范围错误：[{lo}, {hi}]")
            norm.append((lo, hi))
        if not norm:
            raise ValueError("bands 不能为空")
        self.bands = tuple(norm)
        self.band = self.bands[0]   # 向后兼容：旧属性仍指向第一个频段
        self.depth_threshold_db = float(depth_threshold_db)
        self.edge_margin_points = int(edge_margin_points)
        self.weight = float(weight)
        self.depth_weight = float(depth_weight)
        self.depth_negative_weight = float(depth_negative_weight)
        self.depth_negative_margin_db = float(depth_negative_margin_db)
        self.depth_clip_db = float(depth_clip_db)
        self.position_weight = float(position_weight)
        self.position_temperature = float(position_temperature)

        step = 10.0 / (self.n_freq - 1)   # 与数据配置一致：0~10 GHz
        slices = []
        for lo, hi in self.bands:
            i0 = int(round(lo / step))
            i1 = int(round(hi / step))
            if i0 < 0 or i1 > self.n_freq - 1 or i1 <= i0:  # 频段过窄/越界
                raise ValueError(
                    f"频段 [{lo}, {hi}] GHz 超出频率轴或过窄 "
                    f"（索引 {i0}..{i1}，共 {self.n_freq} 点）")
            slices.append((i0, i1 + 1))
        # (n_bands, 2)：每行是一个频段的 [start, end) 切片
        self.register_buffer("_band_slices", torch.tensor(slices, dtype=torch.long),
                             persistent=False)

    def _band_curve(self, x: torch.Tensor, ch: int, band_index: int = 0) -> torch.Tensor:
        i0, i1 = self._band_slices[band_index].tolist()
        return x[:, ch * self.n_freq:(ch + 1) * self.n_freq][:, i0:i1]

    def _interior_depth(self, band_curve: torch.Tensor):
        """
        返回 (depth, pos)：
          depth = relu(min(两端) - min(内部点))
          pos   = 内部最深处在 band 内的索引（0-based）

        用 min(两端) 而不是 max：从带边一路单调下降的曲线不会被误判为
        「带内凹陷」，与 MatchingNetworkReward._interior_depth 一致。
        """
        if band_curve.shape[1] < 3:
            z = torch.zeros(band_curve.shape[0], device=band_curve.device,
                            dtype=band_curve.dtype)
            return z, z.long()
        edge = torch.minimum(band_curve[:, 0], band_curve[:, -1])
        inner_vals, inner_idx = band_curve[:, 1:-1].min(dim=1)
        depth = torch.relu(edge - inner_vals)
        return depth, inner_idx + 1

    def _resonance_mask(self, targets: torch.Tensor) -> torch.Tensor:
        """targets: (B, 3*n_freq) -> (B,) bool

        只要**任一**频段的**任一**通道(S11/S22)存在足够深的带内凹陷，
        该样本就算谐振样本并加权。这样低频段的凹陷样本不会被高频段
        的平坦样本淹没。
        """
        mask = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
        for bi in range(len(self.bands)):
            for ch in (0, 2):   # S11 / S22
                depth, _ = self._interior_depth(self._band_curve(targets, ch, bi))
                mask = mask | (depth >= self.depth_threshold_db)
        return mask

    def _soft_argmin(self, band_curve: torch.Tensor) -> torch.Tensor:
        """可微的期望位置，用于监督凹陷位置。"""
        n = band_curve.shape[1]
        if n < 3:
            return torch.zeros(band_curve.shape[0], device=band_curve.device,
                               dtype=band_curve.dtype)
        logits = -band_curve / max(self.position_temperature, 1e-6)
        prob = torch.softmax(logits, dim=1)
        idx = torch.arange(n, device=band_curve.device, dtype=band_curve.dtype)
        return (prob * idx).sum(dim=1)

    def _depth_loss(self, outputs: torch.Tensor, targets: torch.Tensor):
        """逐频段、逐通道计算深度/位置监督，返回 (depth_loss, position_loss)。

        对每个 band 分别统计正样本（真实有凹陷）和负样本（真实平坦）：
          * 正样本用 SmoothL1 拟合凹陷深度，并可监督谷的位置
          * 负样本用 hinge 抑制虚假凹陷

        每个 band 的贡献等权平均（先算 band 内、再跨 band 平均），
        这样 0.24% 的低频凹陷样本不会被 3.83% 的高频样本淹没。
        """
        depth_per_band = []
        position_per_band = []

        for bi in range(len(self.bands)):
            depth_terms = []
            position_terms = []
            for ch in (0, 2):
                t_band = self._band_curve(targets, ch, bi)
                p_band = self._band_curve(outputs, ch, bi)
                t_depth, t_pos = self._interior_depth(t_band)
                p_depth, _ = self._interior_depth(p_band)

                t_depth = t_depth.clamp(max=self.depth_clip_db)
                p_depth = p_depth.clamp(max=self.depth_clip_db)
                pos_mask = t_depth >= self.depth_threshold_db

                if pos_mask.any():
                    depth_terms.append(F.smooth_l1_loss(
                        p_depth[pos_mask], t_depth[pos_mask], beta=1.0, reduction="mean"))
                    if self.position_weight > 0.0:
                        p_pos = self._soft_argmin(p_band)
                        position_terms.append(F.l1_loss(
                            p_pos[pos_mask], t_pos[pos_mask].to(p_pos.dtype),
                            reduction="mean"))

                neg_mask = ~pos_mask
                if neg_mask.any() and self.depth_negative_weight > 0.0:
                    false_dip = torch.relu(p_depth[neg_mask] - self.depth_negative_margin_db)
                    depth_terms.append(self.depth_negative_weight * false_dip.mean())

            if depth_terms:
                depth_per_band.append(torch.stack(depth_terms).mean())
            if position_terms:
                position_per_band.append(torch.stack(position_terms).mean())

        depth_loss = (torch.stack(depth_per_band).mean()
                      if depth_per_band else outputs.new_zeros(()))
        position_loss = (torch.stack(position_per_band).mean()
                         if position_per_band else outputs.new_zeros(()))
        return depth_loss, position_loss

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # 统一拿到逐样本损失 (B,)。
        #
        # 旧代码写的是 `if per_sample.dim() == 0: per_sample = ...`，
        # 但兜底分支只在标量时触发，而 nn.L1Loss(reduction='mean') 恰好
        # 返回标量 —— 于是 per_sample 是 0 维，后面与 (B,) 的 mask 做
        # torch.where 直接报 "size of tensor a (B) must match b (1)"。
        # 这里改成始终按逐样本计算，不依赖 criterion 的 reduction 设置，
        # 既修了这个 bug，也保证加权逻辑真正生效。
        per_sample = (outputs - targets).abs().reshape(outputs.shape[0], -1).mean(dim=1)

        w = torch.ones_like(per_sample)
        w = torch.where(self._resonance_mask(targets),
                        torch.full_like(w, self.weight), w)
        loss = (per_sample * w).sum() / w.sum()

        if self.depth_weight > 0.0 or self.position_weight > 0.0:
            depth_loss, position_loss = self._depth_loss(outputs, targets)
            loss = loss + self.depth_weight * depth_loss
            if self.position_weight > 0.0:
                loss = loss + self.position_weight * position_loss

        return loss
