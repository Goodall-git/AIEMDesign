import torch

from src.registry import REWARD
import src.reward  # noqa: F401  触发注册

N_FREQ = 101
I0, I1 = 75, 99     # 7.5 ~ 9.9 GHz


def _reward(**kwargs):
    return REWARD.build_from({
        "type_": "MatchingNetworkReward",
        "passband": [7.5, 9.9],
        "freq_start_ghz": 0.0,
        "freq_end_ghz": 10.0,
        "num_points": N_FREQ,
        "s_param_order": ["S11", "S21", "S22"],
        "weight_s21_pass": 0.5,
        "weight_s11_min": 1.0,
        "weight_s22_min": 1.0,
        "min_target_db": -25.0,
        "weight_dip": 0.5,
        "min_edge_headroom_db": 2.0,
        "weight_min_position": 0.5,
        **kwargs,
    })


def _design(dip11: float, dip22: float) -> torch.Tensor:
    """带内两端 -10 dB、中间出现指定深度凹陷的三通道曲线。"""
    s = torch.full((1, 3 * N_FREQ), -10.0)
    s[0, N_FREQ:2 * N_FREQ] = -0.5      # S21 通带
    s[0, I0 + 10] = -10.0 - dip11
    s[0, 2 * N_FREQ + I0 + 12] = -10.0 - dip22
    return s


def test_monotonic_curve_gets_no_matching_credit():
    """从带边一路下降不是带内极小值，匹配项应被门控掉。"""
    r = _reward(weight_dip=0.0, weight_min_position=0.0)
    s = torch.full((1, 3 * N_FREQ), -10.0)
    s[0, I0:I1 + 1] = torch.linspace(-10.0, -40.0, I1 - I0 + 1)
    s[0, N_FREQ:2 * N_FREQ] = -0.5
    s[0, 2 * N_FREQ:I0] = -10.0

    # 只剩 S21 项：0.5 * (-0.5)
    assert torch.allclose(r(s), torch.tensor([-0.25]), atol=1e-5)


def test_deeper_dip_scores_higher():
    r = _reward(weight_min_position=0.0)
    shallow = r(_design(2.5, 2.5))
    deep = r(_design(12.0, 12.0))
    assert deep.item() > shallow.item()


def test_cap_min_prevents_runaway_score():
    r = _reward(weight_dip=0.0, weight_min_position=0.0)
    at_cap = r(_design(15.0, 15.0))       # -25 dB
    beyond_cap = r(_design(40.0, 40.0))   # -50 dB -> 截断到 -25
    assert torch.allclose(at_cap, beyond_cap, atol=1e-5)


def test_max_dip_bonus_caps_bonus():
    r = _reward(weight_min_position=0.0, max_dip_bonus_db=5.0)
    moderate = r(_design(15.0, 15.0))     # 带内最深 -25 dB
    extreme = r(_design(30.0, 30.0))      # 带内最深 -40 dB
    # 匹配项在 -25 dB 已封顶，dip bonus 在 5 dB 封顶 -> 两者相等
    assert torch.allclose(moderate, extreme, atol=1e-5)


# ---------------------------------------------------------------------------
# 带外谐振抑制（weight_out_of_band_penalty）
# ---------------------------------------------------------------------------

def _design_with_out_of_band(dip11: float, out11: float) -> torch.Tensor:
    """
    带内 I0+10 处有 dip11 深凹陷；带外 0.5 GHz 处有 out11 深的凹陷。
    用来模拟「带内副瓣 + 带外主谐振」的错误解。
    """
    s = torch.full((1, 3 * N_FREQ), -10.0)
    s[0, N_FREQ:2 * N_FREQ] = -0.5
    s[0, I0 + 10] = -10.0 - dip11
    s[0, 5] = out11                       # 带外 0.5 GHz
    s[0, 2 * N_FREQ + I0 + 12] = -10.0 - dip11
    s[0, 2 * N_FREQ + 5] = out11
    return s


def test_out_of_band_penalty_defaults_off():
    """默认 weight=0，行为必须与旧版完全一致（向后兼容）。"""
    r0 = _reward(weight_dip=0.0, weight_min_position=0.0)
    r1 = _reward(weight_dip=0.0, weight_min_position=0.0,
                 weight_out_of_band_penalty=0.0)
    s = _design_with_out_of_band(15.0, -30.0)
    assert torch.allclose(r0(s), r1(s))


def test_out_of_band_penalty_punishes_deeper_outside_resonance():
    """带外谐振比带内更深时，必须被惩罚。"""
    off = _reward(weight_dip=0.0, weight_min_position=0.0)
    on = _reward(weight_dip=0.0, weight_min_position=0.0,
                 weight_out_of_band_penalty=2.0)
    bad = _design_with_out_of_band(15.0, -30.0)    # 带内 -25, 带外 -30 => 更差
    assert on(bad).item() < off(bad).item()


def test_out_of_band_penalty_ignores_shallower_outside():
    """带外比带内浅（这是正常情况）不应扣分。"""
    off = _reward(weight_dip=0.0, weight_min_position=0.0)
    on = _reward(weight_dip=0.0, weight_min_position=0.0,
                 weight_out_of_band_penalty=2.0)
    good = _design_with_out_of_band(15.0, -10.0)   # 带内 -25, 带外 -10 => 正常
    assert torch.allclose(off(good), on(good))


def test_out_of_band_penalty_does_not_reward_flat_solutions():
    """
    关键回归：惩罚不能诱导出「整条压平」的退化解。

    曾经把惩罚写成无条件 relu(in_min - out_min)，结果带内不带外都压平的
    版图惩罚恒为 0，反而成了最优解（实测 reward 57 -> -0.44，
    S11 带内只剩 -2.5 dB）。乘上带内门控后，平坦解拿不到任何匹配分。
    """
    on = _reward(weight_dip=0.0, weight_min_position=0.0,
                 weight_out_of_band_penalty=2.0)
    flat = torch.full((1, 3 * N_FREQ), -10.0)      # 全程 -10 dB，没有任何凹陷
    bad = _design_with_out_of_band(15.0, -30.0)
    assert on(bad).item() > on(flat).item()


def test_out_of_band_penalty_scales_with_margin():
    """margin 越大，允许带外更深越多，扣分越少。"""
    strict = _reward(weight_dip=0.0, weight_min_position=0.0,
                     weight_out_of_band_penalty=2.0, out_of_band_margin_db=0.0)
    loose = _reward(weight_dip=0.0, weight_min_position=0.0,
                    weight_out_of_band_penalty=2.0, out_of_band_margin_db=10.0)
    s = _design_with_out_of_band(15.0, -30.0)      # 超出 5 dB
    assert loose(s).item() > strict(s).item()
