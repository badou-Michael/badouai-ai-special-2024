import torch
import torch.nn as nn
from scipy.signal import bilinear

from 成元林.第十六周.Unet.model_parts.DoubleConv import DoubleConv
from 成元林.第十六周.Unet.model_parts.DownConv import DownConv
from 成元林.第十六周.Unet.model_parts.OutputConv import OutputConv
from 成元林.第十六周.Unet.model_parts.UpConv import Upconv


class UnetModel(nn.Module):
    def __init__(self, in_channels, in_classes, bilinear=False):
        super(UnetModel,self).__init__()
        self.in_channels = in_channels
        self.in_classes = in_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(self.in_channels,64)
        self.down1 = DownConv(64,128)
        self.down2 = DownConv(128,256)
        self.down3 = DownConv(256,512)
        self.down4 = DownConv(512,1024)
        self.up1 = Upconv(1024,512,bilinear=bilinear)
        self.up2 = Upconv(512,256,bilinear=bilinear)
        self.up3 = Upconv(256,128,bilinear=bilinear)
        self.up4 = Upconv(128,64,bilinear=bilinear)
        self.outconv = OutputConv(64,in_classes)

    def forward(self,x):
        x1 = self.inc(x) #输出64
        x2 = self.down1(x1) #64-》128
        x3 = self.down2(x2) #128-》256
        x4 = self.down3(x3) # 256=》512
        x5 = self.down4(x4)  #输出1024
        x = self.up1(x5,x4) #1024->512 合并1024 ——》512
        x = self.up2(x,x3)  #合并512-》输出256
        x = self.up3(x,x2) #合并256->输出128
        x = self.up4(x,x1) #合并128 =》输出64
        out = self.outconv(x)
        return out


if __name__ == '__main__':
    net = UnetModel(in_channels=1, in_classes=1)
    print(net)