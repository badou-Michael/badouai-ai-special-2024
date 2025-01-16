import torch
import torch.nn as nn

from 成元林.第十六周.Unet.model_parts.DoubleConv import DoubleConv


class DownConv(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(DownConv,self).__init__()
        self.max_pool2d_and_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels,out_channels)
        )

    def forward(self,x):
        return self.max_pool2d_and_conv(x)