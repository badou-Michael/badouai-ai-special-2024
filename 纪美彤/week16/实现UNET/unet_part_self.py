# 在encode部分，作者遵循了之前卷积网络的结构，就是不断用两个3 x 3卷积核进行卷积，之
# 后用ReLU和2 x 2的pooling层（stride = 2）来降低feature map的大小。每一次down sample都
# 会让相应的feature map的channel数增大一些。
# • 在decode部分，为了和之前encoder的feature map拼接起来，作者用2 x 2的反卷积降低
# feature map的channel数目。拼接之后用1 x 1的卷积核对通道数进行处理就可以了。网络一共
# 有23个卷积层


import torch
import torch.nn as nn
import torch.nn.functional as F

# 由于unet中使用了很多重合的结构，可以先将这些结构实现，再直接调用

class DoubleConv(nn.Module):
    # 两次3*3的卷积,batchnorm,relu
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    # 下采样maxpooling+DoubleConv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    # 先进行copy and crop
    # 上采样DoubleConv+upconv

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
