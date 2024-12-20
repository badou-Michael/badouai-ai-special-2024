import scipy
import numpy as np
import numpy
import matplotlib.pyplot as plt


class diymodel:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate
        self.activation_function = lambda x: scipy.special.expit(x)

        self.wih = np.random.rand(hiddennodes, inputnodes) - 0.5
        self.who = np.random.rand(outputnodes, hiddennodes) - 0.5

    def train(self, inputs_list, targets_list):
        inputs = np.array(inputs_list, ndmin=2).T
        targets = np.array(targets_list, ndmin=2).T
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        #计算误差
        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))
        self.who += self.lr * np.dot(output_errors*final_outputs*(1-final_outputs), hidden_outputs.T)
        self.wih += self.lr * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs), inputs.T)

    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


inputnodes = 784
hiddennodes = 200
outputnodes = 10
learningrate = 0.1
epoch = 10

n = diymodel(inputnodes, hiddennodes, outputnodes, learningrate)

training_data_file = open("NeuralNetWork_从零开始/dataset/mnist_train.csv")
training_data_list = training_data_file.readlines()
training_data_file.close()

for i in range(epoch):
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (numpy.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
        targets = np.zeros(outputnodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)


data_file = open("NeuralNetWork_从零开始/dataset/mnist_test.csv")
test_data_list = data_file.readlines()
data_file.close()
scores = []
sum = 0
for record in test_data_list:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:])/255.0 * 0.99 + 0.01
    a = np.zeros(outputnodes) * 0.01
    a[int(all_values[0])] = 0.99
    targets = a
    b = n.query(inputs)
    label_predict = np.argmax(b)
    if int(all_values[0])==label_predict:
        scores.append(1)
        sum += 1
    else:
        scores.append(0)
print("针对测试集的预测结果如下：")
print(scores)
print(f"测试集准确率：%.2f%%" % (sum/len(scores)*100))
