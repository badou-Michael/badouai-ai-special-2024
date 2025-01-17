import torch
import torch.nn as nn
import torch.nn.functional as F

from 成元林.第十六周.Unet.model_parts.DoubleConv import DoubleConv


class Upconv(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Upconv, self).__init__()
        if bilinear == True:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.double_conv = DoubleConv(in_channels, in_channels//2)
        else:
            self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2,
                                         stride=2)
            self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.double_conv(x)
