#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：unet_parts.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/16 16:13 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class doubleConv(nn.Module):
    """
    双重卷积模块
    包含两次连续的卷积操作，每次卷积后跟BatchNorm和ReLU激活
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            # 第一次卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # 第二次卷积
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """前向传播"""
        return self.double_conv(x)


class Down(nn.Module):
    """
    下采样模块
    包含最大池化和双重卷积操作

    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 2x2最大池化，步长为2
            doubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """前向传播"""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """
    上采样模块
    包含上采样操作和双重卷积
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        bilinear (bool): 是否使用双线性插值进行上采样
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2,
                                  mode='bilinear',
                                  align_corners=True)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels // 2,
                                         in_channels // 2,
                                         kernel_size=2,
                                         stride=2)

        self.conv = doubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        前向传播
        参数:
            x1: 来自上一层的特征图
            x2: 跳跃连接的特征图
        """
        # 对x1进行上采样
        x1 = self.up(x1)

        # 计算特征图尺寸差异
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # 对x1进行padding以匹配x2的尺寸
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 特征图拼接
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outConv(nn.Module):
    """
    输出卷积模块
    使用1x1卷积调整通道数到目标类别数
    参数:
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数（等于类别数）
    """

    def __init__(self, in_channels, out_channels):
        super(outConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """前向传播"""
        return self.conv(x)