import torch
import torch.nn as nn
import torch.optim as optim


# 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 创建模型和优化器
model = SimpleNet()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(10):
    optimizer.zero_grad()
    input_data = torch.randn(1, 10)
    target = torch.randn(1, 1)
    output = model(input_data)
    loss = nn.MSELoss()(output, target)
    loss.backward()
    optimizer.step()
    print("Epoch: {}, Loss: {:.4f}".format(epoch + 1, loss.item()))
