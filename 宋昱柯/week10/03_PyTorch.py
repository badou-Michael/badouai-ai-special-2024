import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:

    def __init__(self, net, cost, optimizer):
        self.net = net
        self.cost = self.create_cost(cost)
        self.optimizer = self.create_optim(optimizer)
        pass

    def create_cost(self, cost):
        support_cost = {"CE": nn.CrossEntropyLoss(), "MSE": nn.MSELoss()}
        return support_cost[cost]

    def create_optim(self, optimizer, **rests):
        support_optim = {
            "SGD": optim.SGD(self.net.parameters(), lr=0.1, **rests),
            "ADAM": optim.Adam(self.net.parameters(), lr=0.01, **rests),
            "RMSP": optim.RMSprop(self.net.parameters(), lr=0.001, **rests),
        }
        return support_optim[optimizer]

    def train(self, train_loader, epochs=5):
        for epoch in range(epochs):
            total_l = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                l = self.cost(outputs, labels)
                l.backward()
                self.optimizer.step()

                total_l += l.item()

                if (i + 1) % 100 == 0:
                    # print(f'epoch {epoch+1},{(i+1)*1./len(train_loader):.2f} loss: {total_l/100:.3f}%')
                    print(
                        "[epoch %d, %.2f%%] loss: %.3f"
                        % (
                            epoch + 1,
                            (i + 1) * 100.0 / len(train_loader),
                            total_l / 100,
                        )
                    )
                    total_l = 0
        print("finish training")

    def evaluate(self, test_loader):
        print("Evalutating...")
        c = 0
        total_c = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.net(inputs)
                predict = torch.argmax(outputs, 1)
                total_c += labels.size(0)
                c += (predict == labels).sum().item()

        print(f"Accuracy: {c/total_c*100:.2f}%")


class Net_01(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


def mnist_data():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                [
                    0,
                ],
                [
                    1,
                ],
            ),
        ]
    )
    trainset = torchvision.datasets.MNIST(
        root="./dataset", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=32, shuffle=True, num_workers=8
    )

    testset = torchvision.datasets.MNIST(
        root="./dataset", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=32, shuffle=False, num_workers=8
    )
    return trainloader, testloader

if __name__ == '__main__':
    net = Net_01().to(device)
    model = Model(net, "CE", "RMSP")
    train_loader, test_loader = mnist_data()
    model.train(train_loader)
    model.evaluate(test_loader)
