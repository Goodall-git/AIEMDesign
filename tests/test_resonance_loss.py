import torch
import torch.nn as nn

from src.registry import LOSS
import src.loss  # noqa: F401  触发注册


N_FREQ = 101
I0, I1 = 75, 99     # 7.5 ~ 9.9 GHz


def _curve_with_dip(channel: int, depth_db: float) -> torch.Tensor:
    """构造一条在带内某点出现指定深度凹陷、其余频点为 -10 dB 的曲线。"""
    y = torch.full((3 * N_FREQ,), -10.0)
    y[I0:I1 + 1] = -10.0
    y[I0 + 12] = -10.0 - depth_db
    return y


def _loss(**kwargs):
    return LOSS.build_from({
        "type_": "ResonanceWeightedLoss",
        "n_freq": N_FREQ,
        "band": [7.5, 9.9],
        "depth_threshold_db": 3.0,
        "edge_margin_points": 1,
        "weight": 12.0,
        "criterion": {"type_": "L1Loss"},
        **kwargs,
    })


def test_resonance_mask_uses_band_edges_not_max():
    """从带边单调下降的曲线不应被判为带内凹陷。"""
    loss = _loss()
    y = torch.full((1, 3 * N_FREQ), -10.0)
    y[0, I0:I1 + 1] = torch.linspace(-10.0, -40.0, I1 - I0 + 1)
    assert loss._resonance_mask(y).item() is False


def test_resonance_mask_detects_true_dip():
    loss = _loss()
    y = torch.stack([_curve_with_dip(0, 8.0)])
    assert loss._resonance_mask(y).item() is True


def test_depth_loss_penalizes_shallow_prediction():
    """真值有深凹陷时，预测太平的曲线应当比预测正确的曲线损失更大。"""
    loss = _loss(depth_weight=2.0, position_weight=0.0)
    target = torch.stack([_curve_with_dip(0, 8.0)])
    flat = torch.full_like(target, -10.0)
    good = target.clone()

    assert loss(flat, target).item() > loss(good, target).item()


def test_depth_loss_suppresses_false_dip_on_flat_target():
    """真值平坦时，预测出深凹陷要额外受罚（防止虚假谐振）。"""
    loss = _loss(depth_weight=2.0, position_weight=0.0,
                 depth_negative_weight=1.0, depth_negative_margin_db=0.5)
    target = torch.full((1, 3 * N_FREQ), -10.0)
    flat = target.clone()
    false_dip = torch.stack([_curve_with_dip(0, 8.0)])

    assert loss(false_dip, target).item() > loss(flat, target).item()


def test_depth_weight_zero_matches_weighted_l1():
    """depth_weight=0 时退化为纯加权 L1，便于与旧实验对照。"""
    loss = _loss(weight=1.0, depth_weight=0.0, position_weight=0.0)
    target = torch.stack([_curve_with_dip(0, 8.0)])
    outputs = target + 1.0
    assert torch.allclose(loss(outputs, target), nn.L1Loss()(outputs, target))


def test_position_loss_reduces_with_correct_position():
    loss = _loss(depth_weight=0.0, position_weight=1.0, position_temperature=0.5)
    target = torch.stack([_curve_with_dip(0, 8.0)])
    wrong = target.clone()
    wrong[0, I0:I1 + 1] = -10.0
    wrong[0, I0 + 4] = -18.0

    assert loss(target, target).item() < loss(wrong, target).item()


# ---------------------------------------------------------------------------
# 多频段（bands）支持
# ---------------------------------------------------------------------------

J0, J1 = 60, 80     # 6.0 ~ 8.0 GHz（低频段）


def _loss_bands(bands, **kwargs):
    return LOSS.build_from({
        "type_": "ResonanceWeightedLoss",
        "n_freq": N_FREQ,
        "bands": bands,
        "depth_threshold_db": 3.0,
        "edge_margin_points": 1,
        "weight": 12.0,
        "criterion": {"type_": "L1Loss"},
        **kwargs,
    })


def _curve_with_low_band_dip(depth_db: float) -> torch.Tensor:
    """只在低频段 6.0~8.0 GHz 有凹陷、高频段平坦的曲线。"""
    y = torch.full((3 * N_FREQ,), -10.0)
    y[J0 + 10] = -10.0 - depth_db
    return y


def test_bands_single_is_backward_compatible():
    """bands=[[a,b]] 应与旧的 band=[a,b] 完全等价。"""
    old = _loss()
    new = _loss_bands([[7.5, 9.9]])
    y = torch.stack([_curve_with_dip(0, 8.0)])
    assert torch.allclose(old(y, y), new(y, y))


def test_bands_accepts_band_kwarg_still():
    """旧写法 band=[a,b] 仍可用，bands 属性应被正确填充。"""
    loss = _loss()
    assert loss.bands == ((7.5, 9.9),)
    assert loss.band == (7.5, 9.9)


def test_bands_slices_are_correct():
    loss = _loss_bands([[7.5, 9.9], [6.0, 8.0]])
    assert loss._band_slices.tolist() == [[75, 100], [60, 81]]


def test_multi_band_catches_low_frequency_dip_that_old_band_misses():
    """
    核心回归测试：低频凹陷样本在旧单 band 配置下被漏掉，多 band 必须命中。
    这正是 6.5~7.5 GHz 凹陷学不出来的根因（0.24% 占比被平坦样本淹没）。
    """
    y = torch.stack([_curve_with_low_band_dip(8.0)])

    old = _loss()                                  # band=[7.5, 9.9]
    new = _loss_bands([[7.5, 9.9], [6.0, 8.0]])

    assert old._resonance_mask(y).item() is False, "旧配置应漏掉低频凹陷"
    assert new._resonance_mask(y).item() is True, "多频段配置应命中低频凹陷"


def test_multi_band_mask_is_union_of_bands():
    """任一频段有凹陷即算谐振样本（取并集，不是交集）。"""
    loss = _loss_bands([[7.5, 9.9], [6.0, 8.0]])
    high = torch.stack([_curve_with_dip(0, 8.0)])      # 只在高频段有凹陷
    low = torch.stack([_curve_with_low_band_dip(8.0)])  # 只在低频段有凹陷
    assert loss._resonance_mask(high).item() is True
    assert loss._resonance_mask(low).item() is True


def test_multi_band_depth_loss_penalizes_flat_low_band():
    """低频段真值有凹陷时，预测成平线必须被惩罚。"""
    loss = _loss_bands([[7.5, 9.9], [6.0, 8.0]],
                       depth_weight=2.0, position_weight=0.0)
    target = torch.stack([_curve_with_low_band_dip(8.0)])
    flat = torch.full_like(target, -10.0)
    assert loss(flat, target).item() > loss(target, target).item()


def test_multi_band_equal_weighting_between_bands():
    """
    两个 band 的深度贡献应等权，低频（样本少）不会被高频淹没。
    构造：只有低频段错、高频段完全正确，损失仍应显著 > 0。
    """
    loss = _loss_bands([[7.5, 9.9], [6.0, 8.0]],
                       depth_weight=2.0, position_weight=0.0)
    target = torch.stack([_curve_with_low_band_dip(10.0)])
    pred = torch.full_like(target, -10.0)
    assert loss(pred, target).item() > 0.5


def test_bands_rejects_invalid_range():
    import pytest
    with pytest.raises(ValueError):
        _loss_bands([[8.0, 7.0]])
    with pytest.raises(ValueError):
        _loss_bands([[7.0, 7.0]])


def test_bands_requires_one_of_band_or_bands():
    """band 和 bands 都不给时应报错，避免静默用错频段。"""
    import pytest
    with pytest.raises(ValueError):
        LOSS.build_from({
            "type_": "ResonanceWeightedLoss",
            "n_freq": N_FREQ,
            "depth_threshold_db": 3.0,
            "criterion": {"type_": "L1Loss"},
        })
