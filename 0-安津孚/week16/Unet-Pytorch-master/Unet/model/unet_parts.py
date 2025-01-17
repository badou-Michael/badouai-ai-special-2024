""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    # 包含两个相同的结构：卷积层（convolution）后接批量归一化层BN和ReLU激活函数
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # nn.Sequential容器，用于按顺序堆叠多个层
        self.double_conv = nn.Sequential(
            # 卷积核大小为3x3，填充（padding）为1
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            # 一个批量归一化层，对输出通道进行归一化处理
            nn.BatchNorm2d(out_channels),
            # 一个ReLU激活函数层，用于引入非线性
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    # 首先是最大池化（Max Pooling），然后是双卷积（DoubleConv）。最大池化用于减小图像的空间尺寸，双卷积用于提取特征。
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # 一个二维最大池化层，池化窗口大小为2x2
            nn.MaxPool2d(2),
            # 减小空间尺寸后的特征图上进行两次卷积操作，提取更丰富的特征
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# nn.Module是构建自定义神经网络层的基础类
class Up(nn.Module):
    # 上采样（Upscaling），然后是双卷积（DoubleConv）
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            # 使用nn.Upsample层进行双线性插值上采样，将图像的空间尺寸放大2倍
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            # 使用nn.ConvTranspose2d层进行转置卷积上采样，将图像的空间尺寸放大2倍
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # 将输入数据x1传递给上采样层self.up，进行上采样操作，恢复图像的空间尺寸
        x1 = self.up(x1)
        # input is CHW
        # 计算上采样后的特征图x1和下采样阶段的特征图x2在高度（Y）和宽度（X）上的尺寸差异。
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        # 使用F.pad函数对x1进行填充，使其高度和宽度与x2相同
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        # 将填充后的特征图x1和下采样阶段的特征图x2在通道维度（dim=1）上进行拼接，融合特征
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
