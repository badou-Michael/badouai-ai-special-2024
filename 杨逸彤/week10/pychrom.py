import torchvision.datasets
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 加载训练数据
def load_data_Mnist():
    predefine = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0,], [1,])
        ]
    )
    trainData = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=predefine)
    trainLoader = torch.utils.data.DataLoader(trainData, batch_size=32, shuffle=True, num_workers=2)

    testData = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=predefine)
    testLoader = torch.utils.data.DataLoader(testData, batch_size=32, shuffle=True, num_workers=2)

    return trainLoader, testLoader

# 创建训练模型
class Model:
    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimizer)

    def create_cost(self, cost):
        choose_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return choose_cost[cost]

    def create_optimizer(self, optimizer):
        choose_optimizer = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001)
        }
        return choose_optimizer[optimizer]

    def train(self, trainLoader, epoches):
        for epoch in range(epoches):
            sum_loss = 0
            for i, data in enumerate(trainLoader):
                inputs, label = data
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.cost(outputs, label)
                loss.backward()
                self.optimizer.step()

                sum_loss += loss.item()
                if i % 100 == 0:
                    print(f'Epoch {epoch+1}, Progress: {((i+1)*100)/len(trainLoader):.2f}%, Loss: {sum_loss/100:.3f}')
                    sum_loss = 0

    def evaluate(self, testLoader):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testLoader:
                inputs, labels = data
                outputs = self.net(inputs)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Accuracy: {accuracy:.2f}%')
        return accuracy

# 定义网络模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 512)
        self.fc4 = torch.nn.Linear(512, 512)
        self.fc5 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return F.softmax(x, dim=1)

# 主程序
if __name__ == '__main__':
    net = Net()
    model = Model(net, 'CROSS_ENTROPY', 'SGD')
    trainLoader, testLoader = load_data_Mnist()
    model.train(trainLoader, epoches=5)
    model.evaluate(testLoader)
