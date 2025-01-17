import torch
import torch.nn as nn
import torch.nn.functional as F

# 构建卷积层
class doubleConv(nn.Module):
    def __init__(self, input_channels, output_channels, mild_channels=None):
        super().__init__()
        if not mild_channels:
            mild_channels = output_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_channels, mild_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mild_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mild_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# 构建池化层进行下采样，步长与kernel大小均为2，w,h为原来一半大小
class downNets(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            doubleConv(input_channels, output_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


# 构建上采样，使用双线性插值进行上采样/使用转置卷积进行上采样
# 指定上采样的比例因子为2，即输出的特征图的高度和宽度都是输入特征图的2倍
class upNets(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up_net = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up_net = nn.ConvTranspose2d(input_channels // 2, input_channels // 2, kernel_size=2, stride=2)
        self.convNet = doubleConv(input_channels, output_channels)

    def forward(self, x1, x2):
        x1 = self.up_net(x1)
        # 计算chw的差异
        diff_h = x2.size()[2] - x1.size()[2]
        diff_w = x2.size()[3] - x1.size()[3]
        # 对上采样的输出进行填充使进行skip_connection时，concat的两个特征图的h,w的大小一致
        x1 = F.pad(x1, [diff_w // 2, diff_w - diff_w // 2,
                        diff_h // 2, diff_h - diff_h // 2])
        if self.training:
            assert x1.size() == x2.size()
        x = torch.concat([x2, x1], dim=1)#在c维度上进行concat
        return self.convNet(x)

# 构建输出网络
class outputConv(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(outputConv, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


# 构建Unet
class Unet(nn.Module):
    def __init__(self, input_channels, output_channels, bilinear=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bilinear = bilinear

        self.inputConv = doubleConv(input_channels, 64)
        self.downConv1 = downNets(64, 128)
        self.downConv2 = downNets(128, 256)
        self.downConv3 = downNets(256, 512)
        self.downConv4 = downNets(512, 512)

        self.upConv1 = upNets(1024, 256, bilinear)
        self.upConv2 = upNets(512, 128, bilinear)
        self.upConv3 = upNets(256, 64, bilinear)
        self.upConv4 = upNets(128, 64, bilinear)
        self.outputC = outputConv(64, output_channels)

    def forward(self, x):
        x1 = self.inputConv(x)
        x2 = self.downConv1(x1)
        x3 = self.downConv2(x2)
        x4 = self.downConv3(x3)
        x5 = self.downConv4(x4)
        x6 = self.upConv1(x5, x4)
        x7 = self.upConv2(x6, x3)
        x8 = self.upConv3(x7, x2)
        x9 = self.upConv4(x8, x1)
        out = self.outputC(x9)
        return out

if __name__=='__main__':
    net = Unet(input_channels=3, output_channels=1)
    print(net)