#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：unet_model.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2025/1/16 10:13
'''

from unet_parts import *


class UNet(nn.Module):
    """
    U-Net模型实现
    用于图像分割任务的端到端深度学习模型

    参数:
        n_channels (int): 输入图像的通道数
        n_classes (int): 分割类别数
        bilinear (bool): 是否使用双线性插值进行上采样，默认为True
    """

    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 下采样路径（编码器）
        self.inc = doubleConv(n_channels, 64)  # 初始卷积，输入通道->64
        self.down1 = Down(64, 128)  # 第一次下采样，64->128
        self.down2 = Down(128, 256)  # 第二次下采样，128->256
        self.down3 = Down(256, 512)  # 第三次下采样，256->512
        self.down4 = Down(512, 512)  # 第四次下采样，512->512

        # 上采样路径（解码器）
        self.up1 = Up(1024, 256, bilinear)  # 第一次上采样，1024->256
        self.up2 = Up(512, 128, bilinear)  # 第二次上采样，512->128
        self.up3 = Up(256, 64, bilinear)  # 第三次上采样，256->64
        self.up4 = Up(128, 64, bilinear)  # 第四次上采样，128->64

        # 输出层
        self.outc = outConv(64, n_classes)  # 最终输出层，64->n_classes

    def forward(self, x):
        """
        前向传播函数
        参数:
            x (tensor): 输入图像张量，形状为 [batch_size, channels, height, width]
        返回:
            logits (tensor): 模型输出的预测结果
        """
        # 编码器路径
        x1 = self.inc(x)  # 初始特征提取
        x2 = self.down1(x1)  # 第一次下采样
        x3 = self.down2(x2)  # 第二次下采样
        x4 = self.down3(x3)  # 第三次下采样
        x5 = self.down4(x4)  # 第四次下采样（瓶颈层）

        # 解码器路径（带跳跃连接）
        x = self.up1(x5, x4)  # 上采样并连接x4特征
        x = self.up2(x, x3)  # 上采样并连接x3特征
        x = self.up3(x, x2)  # 上采样并连接x2特征
        x = self.up4(x, x1)  # 上采样并连接x1特征

        # 输出层
        logits = self.outc(x)  # 生成最终分割图

        return logits