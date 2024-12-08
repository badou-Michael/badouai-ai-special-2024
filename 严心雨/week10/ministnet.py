import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

#数据集
def mnist_load_data():
    #数据预处理
    """
    transforms.Compose():可以将多个变换操作组合在一起。例如，可以将图像的缩放、裁剪和归一化登操作组合在一起，形成一个预处理流程
    :return:
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0,],[1,]) #transforms.Normalize([0,], [1,]) 标准化、归一化
        ]
    )
    trainset = torchvision.datasets.MNIST('.data/',train = True,download = True,transform = transform)
    # shuffle 打乱顺序；num_workers 子进程数量
    trainload = torch.utils.data.DataLoader(trainset,batch_size = 32,shuffle = True,num_workers = 2)

    testset = torchvision.datasets.MNIST('.data/',train = False,download = True,transform = transform)
    testload = torch.utils.data.DataLoader(testset,batch_size = 32,shuffle = True,num_workers =2)
    return trainload,testload

#定义网络结构
class MnistNet(torch.nn.Module):
    """
    init 函数里面放一些需要训练的层
    forward 函数放一些不需要训练的层
    """
    def __init__(self):
        """
        MnistNet子类继承了父类的所有属性和方法，父类属性自然会用父类方法来进行初始化
        """
        super(MnistNet,self).__init__()
        self.fc1 = torch.nn.Linear(28*28,512)
        self.fc2 = torch.nn.Linear(512,512)
        self.fc3 = torch.nn.Linear(512,10)

    def forward(self,x):
        x = x.view(-1,28*28)#把输入的28*28二维的变成了一维的
        """
        F.relu 激活函数
        F.softmax(input,dim=1) 按行Softmax,行和为1（即1维度进行归一化）
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x

class Model(object):
    def __init__(self,net,cost,optimizer):
        self.net = net
        self.cost = self.Create_cost(cost)
        self.optimizer = self.Create_optimizer(optimizer)

    #损失函数
    def Create_cost(self,cost):
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]

    #优化项
    def Create_optimizer(self,optimizer, **rests):
        support_optimizer = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optimizer[optimizer]

    #训练
    def train(self,train_loader,epoches=3):
        for epoch in range(epoches):
            running_loss = 0
            """
            enumerate(iterable,start=0) : 返回枚举对象，返回元组
            用于可迭代/可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标。
            start=0 表示从索引从0开始
            """
            for i,data in enumerate(train_loader,0):
                inputs,lables = data
                """清除所有优化的torch.Tensor的梯度。这是因为梯度是累积计算的，而不是被替换。如果不清零，梯度会与前一个批次的数据相关联，
                   导致每次迭代时梯度不正确。因此，在每次迭代开始前，需要调用此函数将梯度归零
                """
                self.optimizer.zero_grad()
                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.cost(outputs,lables)
                loss.backward()
                self.optimizer.step()

                """
                .item():取一个元素张量里面的具体元素值并返回该值，可以将一个零维张量转换成int型或者float型，
                在计算loss,accuracy时常用到
                """
                running_loss += loss.item()
                """
                i=0,100,200,300,...
                """
                if i % 100 ==0:
                    """ %.2f:将数字四舍五入到小数点后2位
                        %%将一个数转换位百分数
                    """
                    print('[epoch %d,%.2f%%] loss:%.3f' %
                          (epoch + 1,(i+100)*1.0/len(train_loader),running_loss/100))
                    running_loss = 0.0
        print('Finished Training')

    #测试
    def evaluate(self,test_loader):
        print('Evaluating ...')
        correct =0
        total = 0
        with torch.no_grad():# no grad when test and predict
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                predicted = torch.argmax(outputs,1)#只输出最大的一个结果
                """
                labels.size(0):批处理大小，样本数量
                labels.size(1):类别数量
                """
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy of the network on the test images: %d %%' %(100*correct/total))

if __name__ == '__main__':
    #[1]定义网络
    net = MnistNet()
    #损失函数 CROSS_ENTROPY，优化项：RMSP
    model = Model(net,'CROSS_ENTROPY', 'RMSP')
    #[2]准备数据
    train_loader, test_loader = mnist_load_data()
    #[3]把数据送进网络
    model.train(train_loader)
    model.evaluate(test_loader)
