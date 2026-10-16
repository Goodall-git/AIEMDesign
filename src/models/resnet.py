'''
Modified from https://raw.githubusercontent.com/pytorch/vision/v0.9.1/torchvision/models/resnet.py

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''
import sys
import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

from functools import partial
from typing import Dict, Type, Any, Callable, Union, List, Optional

from src.registry import MODEL

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class GAPHead(nn.Module):
    """
    旧版回归头：全局平均池化 + 单层全连接（等价于改造前的 avgpool + fc）。

    保留它的唯一目的是能一键切回旧结构做对照实验（conf 里 head_type: gap）。

    缺点：AdaptiveAvgPool2d((1,1)) 把整个 HxW 版图平均成 C 个数，空间位置信息
    全部丢失。而电磁响应强依赖金属图形"在哪里"——同一个图形平移到不同位置，
    S 参数完全不同；GAP 之后网络只能靠通道去隐式编码位置，代价很高。
    """

    def __init__(self, in_channels, out_dim):
        super(GAPHead, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_channels, out_dim)

    def forward(self, feat):
        x = self.avgpool(feat)
        x = x.flatten(1)
        return self.fc(x)


class SpectralHead(nn.Module):
    """
    新版回归头：保留全部空间位置 + 用 1D 卷积在频率轴上生成结构化输出。

    与 GAPHead 的两点本质区别：

    1. 不做全局平均池化。把 (B, C, H, W) 直接展平成 C*H*W 维再线性投影，
       19*15=285 个空间位置一个都不丢，网络可以自己学"哪块金属在哪个位置"。

    2. 频率轴不是 out_dim 个彼此独立的线性输出，而是按频域序列生成：
         种子向量 + 可学习的频率位置编码 -> (B, hidden, n_freq)
         若干层 1D 卷积在频率方向做平滑与局部建模 -> (B, mid, n_freq)
         1x1 卷积输出三个通道 -> (B, n_s_params, n_freq)
       相邻频点共享卷积核，等于给模型加上了"S 参数曲线沿频率是光滑的"这一先验，
       而不是让 303 个输出各自独立拟合（那样容易出现锯齿）。

    输出仍按 (n_s_params, n_freq) 的行优先顺序展平成 (B, out_dim)，
    所以下游（L1Loss / reward / infer）完全不需要改。
    """

    def __init__(self, in_channels, spatial, out_dim, n_freq=101,
                 hidden=256, mid=128, n_layers=3, kernel_size=5, dropout=0.0):
        super(SpectralHead, self).__init__()
        h, w = spatial
        self.spatial = (int(h), int(w))
        self.n_freq = int(n_freq)

        if out_dim % self.n_freq != 0:
            raise ValueError(
                f"out_dim({out_dim}) 必须是 n_freq({self.n_freq}) 的整数倍，"
                f"即 out_dim = n_s_params * n_freq")
        self.n_s_params = out_dim // self.n_freq
        self.out_dim = out_dim

        # 1) 保留全部空间位置：C*H*W -> hidden
        self.fc = nn.Linear(in_channels * self.spatial[0] * self.spatial[1], hidden)
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 2) 频率位置编码：让每个频点拥有自己的偏置
        self.freq_pos = nn.Parameter(torch.zeros(1, hidden, self.n_freq))
        nn.init.normal_(self.freq_pos, std=0.02)

        # 3) 频率方向的 1D 卷积：局部平滑 + 相邻频点共享权重
        layers = []
        c_in = hidden
        for _ in range(n_layers):
            layers += [
                nn.Conv1d(c_in, mid, kernel_size=kernel_size,
                          padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(mid),
                nn.ReLU(inplace=True),
            ]
            c_in = mid
        self.refine = nn.Sequential(*layers)

        # 4) 输出 (B, n_s_params, n_freq)
        self.out_conv = nn.Conv1d(c_in, self.n_s_params, kernel_size=1)

    def forward(self, feat):
        if feat.shape[2] != self.spatial[0] or feat.shape[3] != self.spatial[1]:
            raise ValueError(
                f"输入特征图空间尺寸 {tuple(feat.shape[2:])} 与 head 配置的 "
                f"spatial={self.spatial} 不一致，请检查 conf 里 model.spatial")

        x = feat.flatten(1)                     # (B, C*H*W)
        x = self.drop(self.act(self.fc(x)))     # (B, hidden)
        x = x.unsqueeze(-1) + self.freq_pos     # (B, hidden, n_freq)
        x = self.refine(x)                      # (B, mid, n_freq)
        x = self.out_conv(x)                    # (B, n_s_params, n_freq)
        return x.flatten(1)                     # (B, out_dim)


class CifarResNet(nn.Module):

    def __init__(self, block, layers, in_channels=1, out_dim=1000,
                 head_type="spectral", spatial=(19, 15), n_freq=101,
                 head_hidden=256, head_mid=128, head_conv_layers=3,
                 head_conv_kernel=5, head_dropout=0.0):
        super(CifarResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(in_channels, 16)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=1)

        feat_channels = 64 * block.expansion
        if head_type == "spectral":
            self.head = SpectralHead(
                in_channels=feat_channels,
                spatial=spatial,
                out_dim=out_dim,
                n_freq=n_freq,
                hidden=head_hidden,
                mid=head_mid,
                n_layers=head_conv_layers,
                kernel_size=head_conv_kernel,
                dropout=head_dropout,
            )
        elif head_type == "gap":
            self.head = GAPHead(feat_channels, out_dim)
        else:
            raise ValueError(f"未知的 head_type={head_type}，可选 spectral / gap")

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return self.head(x)

@MODEL.register
def EMResnet(layers, in_channels, out_dim, head_type="spectral",
             spatial=(19, 15), n_freq=101, head_hidden=256, head_mid=128,
             head_conv_layers=3, head_conv_kernel=5, head_dropout=0.0) -> CifarResNet:
    return CifarResNet(BasicBlock, layers, in_channels, out_dim,
                       head_type=head_type, spatial=spatial, n_freq=n_freq,
                       head_hidden=head_hidden, head_mid=head_mid,
                       head_conv_layers=head_conv_layers,
                       head_conv_kernel=head_conv_kernel,
                       head_dropout=head_dropout)

if __name__ == "__main__":
    # 与项目当前数据一致：3 通道 19x15 版图 -> S11/S21/S22 x 101 频点 = 303
    x = torch.randn(2, 3, 19, 15)

    model = EMResnet([2, 2, 2], in_channels=3, out_dim=303)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"spectral head: output={tuple(model(x).shape)}, params={n_params/1e6:.2f}M")

    legacy = EMResnet([2, 2, 2], in_channels=3, out_dim=303, head_type="gap")
    n_legacy = sum(p.numel() for p in legacy.parameters())
    print(f"gap head     : output={tuple(legacy(x).shape)}, params={n_legacy/1e6:.2f}M")
