# MobileNet网络结构
# 深度可分离卷积
# 深度可分离卷积 分为两层：第一层 深度卷积（Depthwise Convolution）用于滤波，
# 第二层 逐点卷积（Pointwise Convolution）用于合并，这样可以极大地减少 计算量 和 模型大小 ；
# MobileNet 结构建立在深度可分离卷积基础之上，在 深度卷积 和 逐点卷积 之后加入归一化和激活层（BN and ReLU）

import torch
import torch.nn as nn
from torchinfo import summary
# pip install torchinfo


def Depthwise_Separable(in_channels, out_channels, stride, padding):

    model = nn.Sequential(
        # 3 × 3 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=padding, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
        # 1 × 1 点卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    return model


class MobileNet(nn.Module):
    def __init__(self, num_classes=1000, width=1):
        super(MobileNet, self).__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
        )

        self.layers = nn.Sequential(
            Depthwise_Separable(width*32, width*64, 1, 1),
            Depthwise_Separable(width*64, width*128, 2, 1),
            Depthwise_Separable(width*128, width*128, 1, 1),
            Depthwise_Separable(width*128, width*256, 2, 1),
            Depthwise_Separable(width*256, width*256, 1, 1),
            Depthwise_Separable(width*256, width*512, 2, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*512, 1, 1),
            Depthwise_Separable(width*512, width*1024, 2, 1),
            Depthwise_Separable(width*1024, width*1024, 2, 4),
        )

        self.pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(width*1024, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.first_conv(x)
        x = self.layers(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        out = self.softmax(x)
        return out


def test():
    net = MobileNet()
    y = net(torch.randn(1, 3, 224, 224))
    print(y.size())
    summary(net, (1, 3, 224, 224))


if __name__ == '__main__':
    test()
