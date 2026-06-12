#!/usr/bin/env python
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn

'''Wavelet Transform'''
def dwt_init(x):
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    x_LL = x1 + x2 + x3 + x4
    x_HL = -x1 - x2 + x3 + x4
    x_LH = -x1 + x2 - x3 + x4
    x_HH = x1 - x2 - x3 + x4

    return x_LL, x_HL, x_LH, x_HH


# 使用哈尔 haar 小波变换来实现二维离散小波
def iwt_init(x_LL, x_HL, x_LH, x_HH):
    h = torch.zeros(
        [x_LL.size(0), x_LL.size(1), x_LL.size(2) * 2, x_LL.size(3) * 2],
        dtype=x_LL.dtype,
        device=x_LL.device,
    )
    x1 = x_LL / 2
    x2 = x_HL / 2
    x3 = x_LH / 2
    x4 = x_HH / 2
    h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
    h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
    h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
    h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
    return h


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)


class IWT(nn.Module):
    def __init__(self):
        super(IWT, self).__init__()
        self.requires_grad = False

    def forward(self, x_LL, x_HL, x_LH, x_HH):
        return iwt_init(x_LL, x_HL, x_LH, x_HH)


'''CBR'''
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


