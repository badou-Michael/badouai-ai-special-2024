import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )
        
    def forward(self, x):
        return self.conv(x)

class ResBlock(nn.Module):
    def __init__(self, channels, hidden_channels=None):
        super(ResBlock, self).__init__()
        if hidden_channels is None:
            hidden_channels = channels
            
        self.block = nn.Sequential(
            ConvBlock(channels, hidden_channels, 1),
            ConvBlock(hidden_channels, channels, 3)
        )
        
    def forward(self, x):
        return x + self.block(x)

class Darknet53(nn.Module):
    def __init__(self):
        super(Darknet53, self).__init__()
        
        # 初始卷积层
        self.conv1 = ConvBlock(3, 32, 3)
        
        # Downsample 1
        self.conv2 = ConvBlock(32, 64, 3, stride=2)
        self.res1 = nn.Sequential(
            ResBlock(64),
        )
        
        # Downsample 2
        self.conv3 = ConvBlock(64, 128, 3, stride=2)
        self.res2 = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
        )
        
        # Downsample 3
        self.conv4 = ConvBlock(128, 256, 3, stride=2)
        self.res3 = nn.Sequential(
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
            ResBlock(256),
        )
        
        # Downsample 4
        self.conv5 = ConvBlock(256, 512, 3, stride=2)
        self.res4 = nn.Sequential(
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
            ResBlock(512),
        )
        
        # Downsample 5
        self.conv6 = ConvBlock(512, 1024, 3, stride=2)
        self.res5 = nn.Sequential(
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
            ResBlock(1024),
        )
        
    def forward(self, x):
        features = []
        
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)
        
        x = self.conv3(x)
        x = self.res2(x)
        
        x = self.conv4(x)
        x = self.res3(x)
        features.append(x)  # 第一个特征图
        
        x = self.conv5(x)
        x = self.res4(x)
        features.append(x)  # 第二个特征图
        
        x = self.conv6(x)
        x = self.res5(x)
        features.append(x)  # 第三个特征图
        
        return features

def test():
    net = Darknet53()
    x = torch.randn(1, 3, 416, 416)
    features = net(x)
    for feature in features:
        print(feature.shape)

if __name__ == '__main__':
    test() 