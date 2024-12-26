'''
# InceptionV3设计思想
# (1) 分解成小卷积很有效，可以降低参数量，减轻过拟合，增加网络非线性的表达能力。
# (2) 卷积网络从输入到输出，应该让图片尺寸逐渐减小，输出通道数逐渐增加，即让空间结构化，将空间信息转化为高阶抽象的特征信息。
# (3) InceptionModule用多个分支提取不同抽象程度的高阶特征的思路很有效，可以丰富网络的表达能力

# 第一步：定义基础卷积模块
PyTorch提供的有六种基本的Inception模块，分别是InceptionA——InceptionE。
InceptionA 得到输入大小不变，通道数为224+pool_features的特征图。
假如输入为(35, 35, 192)的数据：
第一个branch：
经过branch1x1为带有64个11的卷积核，所以生成第一张特征图(35, 35, 64)；
第二个branch：
首先经过branch5x5_1为带有48个11的卷积核，所以第二张特征图(35, 35, 48)，
然后经过branch5x5_2为带有64个55大小且填充为2的卷积核，特征图大小依旧不变，因此第二张特征图最终为(35, 35, 64)；
第三个branch：
首先经过branch3x3dbl_1为带有64个11的卷积核，所以第三张特征图(35, 35, 64)，
然后经过branch3x3dbl_2为带有96个33大小且填充为1的卷积核，特征图大小依旧不变，因此进一步生成第三张特征图(35, 35, 96)，
最后经过branch3x3dbl_3为带有96个33大小且填充为1的卷积核，特征图大小和通道数不变，因此第三张特征图最终为(35, 35, 96)；
第四个branch：
首先经过avg_pool2d，其中池化核33，步长为1，填充为1，所以第四张特征图大小不变，通道数不变，第四张特征图为(35, 35, 192)，
然后经过branch_pool为带有pool_features个的11卷积，因此第四张特征图最终为(35, 35, pool_features)；
最后将四张特征图进行拼接，最终得到(35，35，64+64+96+pool_features)的特征图。

InceptionB 得到输入大小减半，通道数为480的特征图，
假如输入为(35, 35, 288)的数据：

第一个branch：
经过branch1x1为带有384个33大小且步长2的卷积核，(35-3+20)/2+1=17所以生成第一张特征图(17, 17, 384)；
第二个branch：
首先经过branch3x3dbl_1为带有64个11的卷积核，特征图大小不变，即(35, 35, 64)；
然后经过branch3x3dbl_2为带有96个33大小填充1的卷积核，特征图大小不变，即(35, 35, 96)，
再经过branch3x3dbl_3为带有96个33大小步长2的卷积核，(35-3+20)/2+1=17，即第二张特征图为(17, 17, 96)；
第三个branch：
经过max_pool2d，池化核大小3*3，步长为2，所以是二倍最大值下采样，通道数保持不变，第三张特征图为(17, 17, 288)；
最后将三张特征图进行拼接，最终得到(17(即Hin/2)，17(即Win/2)，384+96+288(Cin)=768)的特征图。

InceptionC 得到输入大小不变，通道数为768的特征图。
假如输入为(17,17, 768)的数据：

第一个branch：
首先经过branch1x1为带有192个1*1的卷积核，所以生成第一张特征图(17,17, 192)；
第二个branch：
首先经过branch7x7_1为带有c7个11的卷积核，所以第二张特征图(17,17, c7)，
然后经过branch7x7_2为带有c7个17大小且填充为03的卷积核，特征图大小不变，进一步生成第二张特征图(17,17, c7)，
然后经过branch7x7_3为带有192个71大小且填充为30的卷积核，特征图大小不变，进一步生成第二张特征图(17,17, 192)，因此第二张特征图最终为(17,17, 192)；
第三个branch：
首先经过branch7x7dbl_1为带有c7个11的卷积核，所以第三张特征图(17,17, c7)，
然后经过branch7x7dbl_2为带有c7个71大小且填充为30的卷积核，特征图大小不变，进一步生成第三张特征图(17,17, c7)，
然后经过branch7x7dbl_3为带有c7个17大小且填充为03的卷积核，特征图大小不变，进一步生成第三张特征图(17,17, c7)，
然后经过branch7x7dbl_4为带有c7个71大小且填充为30的卷积核，特征图大小不变，进一步生成第三张特征图(17,17, c7)，
然后经过branch7x7dbl_5为带有192个17大小且填充为03的卷积核，特征图大小不变，因此第二张特征图最终为(17,17, 192)；
第四个branch：
首先经过avg_pool2d，其中池化核33，步长为1，填充为1，所以第四张特征图大小不变，通道数不变，第四张特征图为(17,17, 768)，
然后经过branch_pool为带有192个的11卷积，因此第四张特征图最终为(17,17, 192)；
最后将四张特征图进行拼接，最终得到(17, 17, 192+192+192+192=768)的特征图。

InceptionD 得到输入大小减半，通道数512的特征图，
假如输入为(17, 17, 768)的数据：
第一个branch：
首先经过branch3x3_1为带有192个11的卷积核，所以生成第一张特征图(17, 17, 192)；
然后经过branch3x3_2为带有320个33大小步长为2的卷积核，(17-3+20)/2+1=8，最终第一张特征图(8, 8, 320)；
第二个branch：
首先经过branch7x7x3_1为带有192个11的卷积核，特征图大小不变，即(17, 17, 192)；
然后经过branch7x7x3_2为带有192个17大小且填充为03的卷积核，特征图大小不变，进一步生成第三张特征图(17,17, 192)；
再经过branch7x7x3_3为带有192个71大小且填充为30的卷积核，特征图大小不变，进一步生成第三张特征图(17,17, 192)；
最后经过branch7x7x3_4为带有192个3*3大小步长为2的卷积核，最终第一张特征图(8, 8, 192)；
第三个branch：

首先经过max_pool2d，池化核大小3*3，步长为2，所以是二倍最大值下采样，通道数保持不变，第三张特征图为(8, 8, 768)；
最后将三张特征图进行拼接，最终得到(8(即Hin/2)，8(即Win/2)，320+192+768(Cin)=1280)的特征图。

InceptionE 最终得到输入大小不变，通道数为2048的特征图。
假如输入为(8,8, 1280)的数据：
第一个branch：
首先经过branch1x1为带有320个11的卷积核，所以生成第一张特征图(8, 8, 320)；
第二个branch：
首先经过branch3x3_1为带有384个11的卷积核，所以第二张特征图(8, 8, 384)，
经过分支branch3x3_2a为带有384个13大小且填充为01的卷积核，特征图大小不变，进一步生成特征图(8,8, 384)，
经过分支branch3x3_2b为带有192个31大小且填充为10的卷积核，特征图大小不变，进一步生成特征图(8,8, 384)，
因此第二张特征图最终为两个分支拼接(8,8, 384+384=768)；
第三个branch：
首先经过branch3x3dbl_1为带有448个11的卷积核，所以第三张特征图(8,8, 448)，
然后经过branch3x3dbl_2为带有384个33大小且填充为1的卷积核，特征图大小不变，进一步生成第三张特征图(8,8, 384)，
经过分支branch3x3dbl_3a为带有384个13大小且填充为01的卷积核，特征图大小不变，进一步生成特征图(8,8, 384)，
经过分支branch3x3dbl_3b为带有384个31大小且填充为10的卷积核，特征图大小不变，进一步生成特征图(8,8, 384)，
因此第三张特征图最终为两个分支拼接(8,8, 384+384=768)；
第四个branch：
首先经过avg_pool2d，其中池化核33，步长为1，填充为1，所以第四张特征图大小不变，通道数不变，第四张特征图为(8,8, 1280)，
然后经过branch_pool为带有192个的11卷积，因此第四张特征图最终为(8,8, 192)；
最后将四张特征图进行拼接，最终得到(8, 8, 320+768+768+192=2048)的特征图。

# 第二步：定义Inceptionv3模块
# 第三步：定义辅助分类器InceptionAux
# 第四步：搭建GoogLeNet网络
# 第五步*：网络结构参数初始化

'''





from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

'''-------------------------第一步：定义基础卷积模块-------------------------------'''

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

    '''-----------------第二步：定义Inceptionv3模块---------------------'''

'''---InceptionA---'''


class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


'''---InceptionB---'''


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


'''---InceptionC---'''


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


'''---InceptionD---'''


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


'''---InceptionE---'''


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


'''-------------------第三步：定义辅助分类器InceptionAux-----------------------'''


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


'''-----------------------第四步：搭建GoogLeNet网络--------------------------'''


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                 inception_blocks=None):
        super(GoogLeNet, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.fc = nn.Linear(2048, num_classes)

    '''输入(229,229,3)的数据，首先归一化输入，经过5个卷积，2个最大池化层。'''

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        '''然后经过3个InceptionA结构，
        1个InceptionB，3个InceptionC，1个InceptionD，2个InceptionE，
        其中InceptionA，辅助分类器AuxLogits以经过最后一个InceptionC的输出为输入。
        '''
        # 35 x 35 x 192
        x = self.Mixed_5b(x)  # InceptionA(192, pool_features=32)
        # 35 x 35 x 256
        x = self.Mixed_5c(x)  # InceptionA(256, pool_features=64)
        # 35 x 35 x 288
        x = self.Mixed_5d(x)  # InceptionA(288, pool_features=64)
        # 35 x 35 x 288
        x = self.Mixed_6a(x)  # InceptionB(288)
        # 17 x 17 x 768
        x = self.Mixed_6b(x)  # InceptionC(768, channels_7x7=128)
        # 17 x 17 x 768
        x = self.Mixed_6c(x)  # InceptionC(768, channels_7x7=160)
        # 17 x 17 x 768
        x = self.Mixed_6d(x)  # InceptionC(768, channels_7x7=160)
        # 17 x 17 x 768
        x = self.Mixed_6e(x)  # InceptionC(768, channels_7x7=192)
        # 17 x 17 x 768
        if self.training and self.aux_logits:
            aux = self.AuxLogits(x)  # InceptionAux(768, num_classes)
        # 17 x 17 x 768
        x = self.Mixed_7a(x)  # InceptionD(768)
        # 8 x 8 x 1280
        x = self.Mixed_7b(x)  # InceptionE(1280)
        # 8 x 8 x 2048
        x = self.Mixed_7c(x)  # InceptionE(2048)

        '''进入分类部分。
        经过平均池化层+dropout+打平+全连接层输出'''
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        x = torch.flatten(x, 1)  # Flatten（）就是将2D的特征图压扁为1D的特征向量，是展平操作，进入全连接层之前使用,类才能写进nn.Sequential
        # N x 2048
        x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    def forward(self, x):
        x, aux = self._forward(x)
        return x, aux

    '''-----------------------第五步：网络结构参数初始化--------------------------'''

    # 目的：使网络更好收敛，准确率更高
    def _initialize_weights(self):  # 将各种初始化方法定义为一个initialize_weights()的函数并在模型初始后进行使用。

        # 遍历网络中的每一层
        for m in self.modules():
            # isinstance(object, type)，如果指定的对象拥有指定的类型，则isinstance()函数返回True

            '''如果是卷积层Conv2d'''
            if isinstance(m, nn.Conv2d):
                # Kaiming正态分布方式的权重初始化
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                '''判断是否有偏置：'''
                # 如果偏置不是0，将偏置置成0，对偏置进行初始化
                if m.bias is not None:
                    # torch.nn.init.constant_(tensor, val)，初始化整个矩阵为常数val
                    nn.init.constant_(m.bias, 0)

                '''如果是全连接层'''
            elif isinstance(m, nn.Linear):
                # init.normal_(tensor, mean=0.0, std=1.0)，使用从正态分布中提取的值填充输入张量
                # 参数：tensor：一个n维Tensor，mean：正态分布的平均值，std：正态分布的标准差
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


'''---------------------------------------显示网络结构-------------------------------'''
from torchsummary import summary
# pip install torch-summary

if __name__ == '__main__':
    net = GoogLeNet(1000)

    summary(net, (3, 299, 299))

