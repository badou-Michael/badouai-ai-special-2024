import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import torchvision.transforms as transforms

class Model():
    def __init__(self, net, cost, optimizer):
        #模型初始化输入网络结构，损失函数，以及优化器
        self.net = net
        self.cost = self.cost_function(cost)
        self.optimizer = self.optim_function(optimizer, 0.0005)

    def cost_function(self, cost):
        support_cost = {
            'CROSS_ENRTOPY': nn.CrossEntropyLoss(),
            'MSE': nn.MSELoss()
        }
        return support_cost[cost]
    
    def optim_function(self, optimizer, lr):
        support_optim = {
            'SGD': optim.SGD(self.net.parameters(),lr),
            'ADAM':optim.Adam(self.net.parameters(), lr),
            'RMSP':optim.RMSprop(self.net.parameters(), lr)
        }
        return support_optim[optimizer]
    
    def train(self, train_loader, epoch):
        for e in range(epoch):
            running_loss = 0
            for train_step, data in enumerate(train_loader):
                inputs, labels = data
                
                # 梯度清零
                self.optimizer.zero_grad()

                #正向传播
                outputs = self.net(inputs)

                #计算loss
                loss = self.cost(outputs, labels)

                #反向传播
                loss.backward()

                #更新参数
                self.optimizer.step()
                running_loss += loss.item()
                if train_step%100 == 0:
                    print('[epcoh:%d  %.2f%% ] loss: %.3f' %(e+1, (train_step + 1)*100./len(train_loader), running_loss/100))
                    running_loss = 0
        print('Training Over!!!!!!!!')
    
    def evaluate(self, test_loader):
        print('Start Evaluating>>>')
        correct_nums = 0
        total_nums = 0
        wrong_nums = 0
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                test_input, test_label = data
                test_out = self.net(test_input)
                res = torch.argmax(test_out, dim=1)
                total_nums += test_label.size(0)
                correct_nums += (res == test_label).sum().item()
                wrong_nums += sum((res != test_label)).item()
        print('Acuarcy:%d%%'%(correct_nums/total_nums*100))
        print('LOSS:%d%%'%(wrong_nums/total_nums*100))
        print(correct_nums, total_nums,wrong_nums)

class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512,10)
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x
    
def mnistdata_loader():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.],[1.])
    ])
    train_set = torchvision.datasets.MNIST(root= './data', train=True, transform= transform, download=True)
    tran_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    test_set = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)
    return tran_loader, test_loader
    # return train_set, test_set


if __name__ == '__main__' :
    net = Mnist()
    train_loader, test_loader = mnistdata_loader()
    # print(test_loader.data.shape)
    model = Model(net, 'CROSS_ENRTOPY', 'RMSP')
    model.train(train_loader,5)
    model.evaluate(test_loader)




        
            

            
