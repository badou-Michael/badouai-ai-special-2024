import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2

class Model:
    def __init__(self, net, cost, optimist):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optimizer(optimist)
        pass

    def create_cost(self, cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }

        return support_cost[cost]

    def create_optimizer(self, optimist, **rests):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP':optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }

        return support_optim[optimist]

    def train(self, train_loader, epoches=10):
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, (i + 1)*1./len(train_loader)*100, running_loss / 100))
                    running_loss = 0.0

        print('Finished Training')

    def evaluate(self, test_loader):
        print('Evaluating ...')
        correct = 0
        total = 0
        with torch.no_grad():  # no grad when test and predict
            for data in test_loader:
                images, labels = data

                outputs = self.net(images)
                predicted = torch.argmax(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

    def eval_one_picture(self,path):
        # 1. 读取图片并转换为灰度图
        image = cv2.imread(path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Shape: (height, width)

        # 2. 调整图像大小到 MNIST 模型的输入大小 (28x28)
        resized_image = cv2.resize(gray_image, (28, 28))  # Resize to (28, 28)

        # 3. 归一化图像数据到 [0, 1]
        normalized_image = resized_image / 255.0

        # 4. 转换为 PyTorch 张量，并添加批次维度
        # 模型的输入需要是四维张量，形状为 (batch_size, channels, height, width)。
        input_tensor = torch.tensor(normalized_image, dtype=torch.float32)  # Shape: (28, 28)
        #input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)

        # 5. 将数据输入到模型中并获取预测结果
        with torch.no_grad():  # 评估时不需要梯度计算
            output = self.net(input_tensor)  # Shape: (1, 10)
            predicted = torch.argmax(output, 1).item()  # 获取预测类别

        # 6. 打印预测结果
        print(f'{path}识别的结果是：{predicted}')

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])
    # torchvision 提供了用于加载和处理常见数据集的工具，这里使用的是 MNIST
    # transform=transform 表示对数据进行预处理
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
    # DataLoader 是 PyTorch 提供的工具，用于将数据集加载为可迭代的数据批次
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,shuffle=True, num_workers=2)
    return trainloader, testloader

class MnistNet(torch.nn.Module):
    def __init__(self):
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28*28, 512)
        self.fc2 = torch.nn.Linear(512, 512)
        self.fc3 = torch.nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x

if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
    model.eval_one_picture('data/my_own_2.png')
    model.eval_one_picture('data/my_own_3.png')
    model.eval_one_picture('data/my_own_4.png')


