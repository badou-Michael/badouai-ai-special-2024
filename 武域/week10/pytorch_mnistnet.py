import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

# Wrapper class for training, evaluating, and managing models
class Model:
    def __init__(self, net, cost, optimist):
        """
        Initialize the Model class.
        Args:
            net: The neural network architecture.
            cost: The cost function to be used ('CROSS_ENTROPY' or 'MSE').
            optimist: The optimizer to be used ('SGD', 'ADAM', or 'RMSP').
        """
        self.net = net
        self.cost = self.create_cost(cost)  # Create cost function
        self.optimizer = self.create_optimizer(optimist)  # Create optimizer

    def create_cost(self, cost):
        """
        Create the cost/loss function.
        Args:
            cost: Name of the cost function.
        Returns:
            The initialized cost function.
        """
        support_cost = {
            'CROSS_ENTROPY': nn.CrossEntropyLoss,  # For classification tasks
            'MSE': nn.MSELoss  # For regression tasks
        }
        return support_cost[cost]()  # Instantiate the loss function

    def create_optimizer(self, optimist, **rests):
        """
        Create the optimizer.
        Args:
            optimist: Name of the optimizer.
            **rests: Additional parameters for the optimizer.
        Returns:
            The initialized optimizer.
        """
        support_optimizer = {
            'SGD': optim.SGD(self.net.parameters(), lr=0.1, **rests),
            'ADAM': optim.Adam(self.net.parameters(), lr=0.01, **rests),
            'RMSP': optim.RMSprop(self.net.parameters(), lr=0.001, **rests)
        }
        return support_optimizer[optimist]

    def train(self, train_loader, epoches=3):
        """
        Train the model.
        Args:
            train_loader: Data loader for training data.
            epoches: Number of training epochs.
        """
        for epoch in range(epoches):
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):
                inputs, labels = data

                self.optimizer.zero_grad()  # Clear gradients
                outputs = self.net(inputs)  # Forward pass
                loss = self.cost(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights

                running_loss += loss.item()
                if i % 100 == 0:  # Print progress every 100 batches
                    print('[epoch %d, %.2f%%] loss: %.3f' %
                          (epoch + 1, 100 * (i + 1) / len(train_loader), running_loss / 100))
                    running_loss = 0.0

        print("Finished Training")

    def evaluate(self, test_loader):
        """
        Evaluate the model.
        Args:
            test_loader: Data loader for testing data.
        """
        print("Evaluating...")
        correct = 0
        total = 0
        with torch.no_grad():  # Disable gradient computation for evaluation
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)  # Forward pass
                predicted = torch.argmax(outputs, 1)  # Get predicted labels
                total += labels.size(0)  # Total samples
                correct += (predicted == labels).sum().item()  # Correct predictions

        print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))

# Function to load MNIST dataset
def load_data():
    """
    Load MNIST dataset.
    Returns:
        trainloader: DataLoader for training data.
        testloader: DataLoader for testing data.
    """
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])])  # Normalize to [-1, 1]

    # Download and prepare training set
    trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                           download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                              shuffle=True, num_workers=2)

    # Download and prepare test set
    testset = torchvision.datasets.MNIST(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                             shuffle=True, num_workers=2)
    return trainloader, testloader

# Define the MNIST neural network
class MnistNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize the MnistNet model.
        """
        super(MnistNet, self).__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 512)  # First fully connected layer
        self.fc2 = torch.nn.Linear(512, 512)  # Second fully connected layer
        self.fc3 = torch.nn.Linear(512, 10)  # Output layer (10 classes)

    def forward(self, x):
        """
        Define the forward pass.
        Args:
            x: Input tensor.
        Returns:
            Output tensor after passing through the network.
        """
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = F.relu(self.fc1(x))  # Apply ReLU to the first layer
        x = F.relu(self.fc2(x))  # Apply ReLU to the second layer
        x = self.fc3(x)  # Raw logits for CrossEntropyLoss
        return x

# Main function to train and evaluate the network
if __name__ == '__main__':
    # Initialize the network
    net = MnistNet()

    # Create the model with the network, cost function, and optimizer
    model = Model(net, 'CROSS_ENTROPY', 'RMSP')

    # Load the data
    train_loader, test_loader = load_data()

    # Train the model
    model.train(train_loader)

    # Evaluate the model
    model.evaluate(test_loader)