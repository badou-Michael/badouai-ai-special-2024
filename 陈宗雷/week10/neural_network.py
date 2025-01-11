#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Name    :neural_network.py
@Desc    : neural network implement
@Time    :2024/12/04 13:41:25
@Author    :chung rae
@Version    :1.0
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
from torch import nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class NeuralNetwork:
    def __init__(
        self,
        inputs: int,
        hidden: int,
        outputs: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        epoch: int = 5,
        learning_rate: float = 0.1,
    ) -> None:
        """
        :param inputs: input layer count
        :param hidden: hidden layer count
        :param outputs: output layer count
        :param train_data
        :param test_data
        :param epoch defaults to 5
        :param learning_rate: learning_rate defaults to 0.1
        """

        self.inputs = inputs
        self.hidden = hidden
        self.outputs = outputs
        self.train_dataset = self.normalization(train_data)
        self.test_dataset = self.normalization(test_data)
        self.learning_rate = learning_rate
        self.epoch = epoch

        # activation function sigmoid

        self.activation_function = lambda x: 1 / (1 + np.exp(-x))

        # input layer and hidden layer weight matrix 'WX+b' of W
        self.ihw = np.random.normal(
            0.0, pow(self.hidden, -0.5), (self.hidden, self.inputs)
        )

        # hidden layer and output layer weight matrix 'WX+b' of W
        self.how = np.random.normal(
            0.0, pow(self.outputs, -0.5), (self.outputs, self.hidden)
        )

    def __train(self, inputs: pd.Series, target: np.ndarray):
        """
        :param inputs: input data
        :param target: target data
        """

        # transform 2d matrix

        inputs = np.array(inputs, ndmin=2).T
        target = np.array(target, ndmin=2).T

        # compute hidden layer input
        hit = np.dot(self.ihw, inputs)

        # compute hidden layer output
        hot = self.activation_function(hit)

        # compute output layer input
        oit = np.dot(self.how, hot)
        # compute output layer output
        oot = self.activation_function(oit)

        # compute loss
        out_loss = oot - target

        hidden_loss = np.dot(self.how.T, out_loss * oot * (1 - oot))

        # update weight

        self.how += self.learning_rate * np.dot(
            out_loss * oot * (1 - oot), np.transpose(hot)
        )

        self.ihw += self.learning_rate * np.dot(
            hidden_loss * hot * (1 - hot), np.transpose(inputs)
        )

    def query(self, inputs: np.ndarray) -> np.ndarray:
        """get out layer output

        :param inputs: input layer inputs
        :return: output layer outputs
        """

        # hidden layer inputs
        hit = np.dot(self.ihw, inputs)
        # hidden layer outputs
        hot = self.activation_function(hit)

        # output layer inputs
        oit = np.dot(self.how, hot)

        # output layer outputs
        oot = self.activation_function(oit)

        return oot


    @staticmethod
    def normalization(data: pd.DataFrame) -> pd.DataFrame:
        """
        @param data:
        @return:
        """
        new_data = data.iloc[:, 1:].apply(lambda x: x / 255 * 0.99 + 0.01)
        return pd.concat([data.iloc[:, :1], new_data], axis=1)

    def train(self):
        """train train_dataset"""
        # train data

        for _ in range(self.epoch):
            for _, row in self.train_dataset.iterrows():
                targets = np.zeros(self.outputs) + 0.01
                targets[int(row[0])] = 0.99
                self.__train(row[1:], targets)

    def evaluate(self) -> np.float64:
        """evaluate test_dataset"""

        result = [
            1 if row[0] == np.argmax(self.query(row[1:])) else 0
            for _, row in self.test_dataset.iterrows()
        ]

        accurate = np.mean(result)

        return accurate

class MinstNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # input layer
        self.fc1 = nn.Linear(28*28, 512)
        # hidden layer
        self.fc2 = nn.Linear(512, 512)
        # output layer
        self.fc3 = nn.Linear(512, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.view(-1, 28*28)
        inputs = F.relu(self.fc1(inputs))
        inputs = F.relu(self.fc2(inputs))
        inputs = F.softmax(self.fc3(inputs), dim=1)
        return inputs

class TorchNetwork:
    def __init__(
            self,
            network: MinstNetwork,
            train_data: DataLoader,
            test_data: DataLoader,
            epoch: int = 5,
            loss_func_type: str="CROSS_ENTROPY",
            optimizer_type: str="RMSP"
    ) -> None:
        """

        @param network:
        @param train_data:
        @param test_data:
        @param epoch: defaults to 5
        @param loss_func_type: defaults to "CROSS_ENTROPY"
        @param optimizer_type: defaults to "RMSP"
        """
        self.network = network
        self.train_dataset = train_data
        self.test_dataset = test_data
        self.loss_function = nn.MSELoss() if loss_func_type == "MSE" else nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.network.parameters(), lr=0.1) if optimizer_type == "SGD" \
            else  optim.Adam(self.network.parameters(), lr=0.01) if optimizer_type == "Adam" \
            else optim.RMSprop(self.network.parameters(), lr=0.001)

        self.epoch = epoch

    def train(self):
        for _ in range(self.epoch):
            running_loss = 0.0
            for i, data in enumerate(self.train_dataset):
                inputs, targets = data
                self.optimizer.zero_grad()
                outputs = self.network(inputs)
                loss = self.loss_function(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if running_loss % 100 == 0:
                    print(f"epoch {self.epoch} batch {i} loss {running_loss / 100}")
                    running_loss = 0.0
        print("Finished Training")

    def evaluate(self) -> float:
        correct, total = 0, 0
        with torch.no_grad():
            for i, data in enumerate(self.test_dataset):
                inputs, targets = data
                outputs = self.network(inputs)
                pred = outputs.argmax(dim=1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()

        accuracy = correct / total
        print(f"Accuracy of net work on test is: {accuracy}")
        return accuracy



class TFNetwork:
    def __init__(self,
                 train_data: tuple[np.ndarray, np.ndarray],
                 test_data: tuple[np.ndarray, np.ndarray],
                 epoch: int = 5,
                 batch: int = 128
                 ):
        """
        @param train_data:
        @param test_data:
        @param epoch: defaults to 5
        @param batch: defaults to 128
        """
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(512, activation='relu', input_shape=(28*28,)))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        model.compile(loss='categorical_crossentropy')
        self.model = model
        self.train_dataset, self.train_targets = train_data
        self.test_dataset, self.test_targets = test_data
        self.train_dataset = self.normalization(self.train_dataset)
        self.test_dataset = self.normalization(self.test_dataset, True)
        self.train_targets = tf.keras.utils.to_categorical(self.train_targets)
        self.test_targets = tf.keras.utils.to_categorical(self.test_targets)
        self.epoch = epoch
        self.batch = batch

    @staticmethod
    def normalization(data: np.ndarray, is_test: bool=False) -> np.ndarray:
        if not is_test:
            data = data.reshape((60000, 28*28)).astype('float32') / 255

        else:
            data = data.reshape((10000, 28 * 28)).astype('float32') / 255

        return data

    def train(self) -> None:
        self.model.fit(self.train_dataset, self.train_targets, epochs=self.epoch, batch_size=self.batch)

    def evaluate(self) -> float:
        _, accuracy = self.model.evaluate(self.test_dataset, self.test_targets, verbose=1)
        return accuracy


def main(network_type: str= "neural"):

    if network_type not in ["neural", "torch", "tf"]:
        network_type = "neural"

    if network_type == "neural":
        train_data = pd.read_csv(
            "./dataset/mnist_train.csv",
            header=None,
            dtype=np.float32,
        )
        test_data = pd.read_csv(
            "./dataset/mnist_test.csv",
            header=None,
            dtype=np.float32,
        )
        model = NeuralNetwork(784, 200, 10, train_data, test_data, epoch=5)

    elif network_type == "torch":
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1.0)])
        train_dataset = datasets.MNIST("./data", train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST("./data", train=False, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        model = TorchNetwork(MinstNetwork(), train_loader, test_loader)


    else:
        (train, train_target), (test, test_target) = tf.keras.datasets.mnist.load_data()
        model = TFNetwork((train, train_target),  (test, test_target))


    model.train()
    accuracy = model.evaluate()

    print(accuracy)

if __name__ == "__main__":

    main("torch")