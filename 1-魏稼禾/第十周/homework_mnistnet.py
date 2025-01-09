import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np

class Model:
    def __init__(self, net, cost, optim):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optim = self.create_optimizer(net, optim)
        
    def create_cost(self, cost):
        cost_dict = {
            "CROSS_ENTROPY":nn.CrossEntropyLoss(),
            "MSE":nn.MSELoss()
        }
        return cost_dict[cost]
    
    def create_optimizer(self, net, optim):
        optim_dict = {
            "SGD":torch.optim.SGD(net.parameters(), lr=0.1),
            "ADAM":torch.optim.Adam(net.parameters(), lr=0.001),
            "RMSP":torch.optim.RMSprop(net.parameters(), lr=0.001)
        }
        return optim_dict[optim]
    
    def train(self, train_loader, epoch_num=5):
        for epoch in range(epoch_num):
            epoch = epoch+1
            print("epoch %d begin"%(epoch))
            self.net.train()
            watch_loss = []
            for index, batch_data in enumerate(train_loader):
                input_ids, labels = batch_data
                
                self.optim.zero_grad()
                preds = self.net(input_ids)
                loss = self.cost(preds, labels)
                loss.backward()
                self.optim.step()
                
                watch_loss.append(loss.item())
                if index%(int(len(train_loader)/2)) == 0:
                    print("avg loss: %f"%(np.mean(watch_loss)))
        print("Finished Training")
        
    def evaluate(self, test_loader):
        self.net.eval()
        with torch.no_grad():
            total = 0
            correct = 0
            for batch_data in test_loader:
                input_ids, targets = batch_data
                preds = self.net(input_ids)
                preds = torch.argmax(preds, dim=-1)
                total += input_ids.shape[0]
                correct += (preds == targets).sum().item()
            print("acc is %f"%(correct/(total+1e-7)))
                
                
class MnistNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        
        return x
    
def mnist_load_data():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0,],[1,])]
    )
    trainset = torchvision.datasets.MNIST(root="./data", train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, 
                                              shuffle=True, num_workers=2)
    testset = torchvision.datasets.MNIST(root="./data", train=False,
                                         download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=False, num_workers=2)
    return trainloader, testloader

if __name__ == "__main__":
    trainloader,testloader = mnist_load_data()
    net = MnistNet()
    model = Model(net, "CROSS_ENTROPY", "ADAM")
    model.train(trainloader)
    model.evaluate(testloader)