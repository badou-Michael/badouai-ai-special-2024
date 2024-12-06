import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

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

    def train(self, train_loader, epoches=3):
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
                          (epoch + 1, (i + 1)*1./len(train_loader), running_loss / 100))
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
                predicted = torch.argmax(outputs, 1) #torch.argmax函数用于返回输入张量中每行（或指定维度）的最大值的索引
                #1：这是dim参数，指定在哪个维度上操作。在这里，1表示沿着列（即每个样本的类别得分）找到最大值的索引。
                total += labels.size(0)
                correct += (predicted == labels).sum().item()##因此，(predicted == labels).sum()会计算所有True值的总和，即计算预测正确的样本数量
                # 。结果是一个标量张量（只有一个元素的张量）。item()方法将张量中的单个值转换为Python的标量值

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,], [1,])])

    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                            download=True, transform=transform)
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


        '''
self.fc1 = torch.nn.Linear(28*28, 512)：
这里定义了第一个全连接层fc1。输入特征数量是28*28，因为假设输入数据是28x28像素的图像，展平后得到784个特征。输出特征数量是512，这意味着该层将输入的784个特征映射到512个特征。
self.fc2 = torch.nn.Linear(512, 512)：
这里定义了第二个全连接层fc2。输入特征数量是512，与前一层的输出特征数量相同。输出特征数量也是512，这意味着该层保持特征数量不变，但对特征进行进一步的线性变换。
self.fc3 = torch.nn.Linear(512, 10)：
这里定义了第三个全连接层fc3。输入特征数量是512，与前一层的输出特征数量相同。输出特征数量是10，这意味着该层将输入的512个特征映射到10个特征，通常对应于10个类别（如MNIST手写数字识别任务中的10个数字类别）。
'''
##MnistNet类定义了一个简单的三层全连接神经网络。
    # 它继承自torch.nn.Module，并在构造函数中定义了三个全连接层：第一个层将28x28的图像展平为一个向量，并映射到512个节点
    # ，第二个层将512个节点映射到另一个512个节点，最后一个层将512个节点映射到10个节点，对应于MNIST数据集中的10个类别。
    def forward(self, x):
        x = x.view(-1, 28*28)
        #x.view(-1, 28*28)：将输入数据x重塑为一个二维张量。-1表示自动计算该维度的大小，
        # 以保持数据的总元素个数不变。28*28表示每个样本被展平为一个784维的向量。这通常用于处理图像数据，例如MNIST数据集中的28x28像素图像。
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x##返回最终的输出张量x，它包含了每个样本属于每个类别的概率。
    #dim=1：指定在哪个维度上计算softmax。对于多分类问题，通常在最后一个维度上计算（dim=1），因为这个维度通常代表类别

    #F是torch.nn.functional模块的别名。这个模块包含了许多用于神经网络的函数，如激活函数、损失函数等。F.relu是这个模块中的一个函数，
    # 用于计算ReLU（Rectified Linear Unit）激活函数。  import torch.nn.functional as F ；ReLU(x)=max(0,x)

    ##F.softmax 将每个样本的 logits 转换为概率分布，使得每个样本的所有类别概率之和为1。
##forward方法定义了数据通过网络的前向传播路径。输入的图像首先被展平，然后通过两个ReLU激活的全连接层，最后通过一个softmax层输出每个类别的概率分布。
if __name__ == '__main__':
    # train for mnist
    net = MnistNet()
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')
    train_loader, test_loader = mnist_load_data()
    model.train(train_loader)
    model.evaluate(test_loader)
