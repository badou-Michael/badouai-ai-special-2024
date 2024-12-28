import torch


class Linear(torch.nn.Module):
    # bias是否包含偏置项
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        # 初始化权重矩阵weight，使用正态分布随机生成out_features行in_features列的值 矩阵包装成torch.nn.Parameter对象
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            # 使用正态分布随机生成out_features个值，向量包装成torch.nn.Parameter对象
            self.bias = torch.nn.Parameter(torch.randn(out_features))

    # 前向传播函数，它接收输入x
    def forward(self, x):
        # 矩阵乘法计算输入x和权重weight的乘积
        x = x.mm(self.weight)
        if self.bias:
            # 将偏置向量bias加到上一步的结果上，expand_as(x)确保偏置向量的形状与x相同
            x = x + self.bias.expand_as(x)
        return x


if __name__ == '__main__':
    # train for mnist
    net = Linear(3, 2)
    x = net.forward
    print('11', x)
