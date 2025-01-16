import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learning_rate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learning_rate
        
        # 初始化权重矩阵
        # shape:(200, 784)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), size=(self.hnodes, self.inodes))
        # shape:(10, 200)
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), size=(self.onodes, self.hnodes))
        
        # 定义激活函数
        self.activationfunc = lambda x: scipy.special.expit(x)
        
    def train(self, inputs, targets):
        inputs = np.array(inputs, ndmin=2).T    # shape:(784, 1)
        targets = np.array(targets, ndmin=2).T  # shape:(10,1)
        
        hidden_inputs = np.dot(self.wih, inputs)  # shape:(200,1)
        hidden_outputs = self.activationfunc(hidden_inputs) # shape:(200,1)
        
        final_inputs = np.dot(self.who, hidden_outputs)   # shape:(10,1)
        final_outputs = self.activationfunc(final_inputs)   # shape:(10,1)
        
        output_errors = targets - final_outputs # shape:(10,1)
        hidden_errors = np.dot(self.who.T, output_errors*final_outputs*(1-final_outputs))   # shape:(200,1)
        
        self.who += self.lr*np.dot(output_errors*final_outputs*(1-final_outputs),
                                   np.transpose(hidden_outputs))
        self.wih += self.lr*np.dot(hidden_errors*hidden_outputs*(1-hidden_outputs),
                                   np.transpose(inputs))
    
    def query(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activationfunc(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activationfunc(final_inputs)
        # print(final_outputs)
        return final_outputs
    
input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
model = NeuralNetWork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#读入训练数据
training_data_file = open("dataset/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()

#训练轮数为5
epoch_num = 5
for _ in range(epoch_num):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = np.asfarray(all_values[1:])/255.0*0.99+0.01
        targets = np.zeros(output_nodes)+0.01
        targets[int(all_values[0])] = 0.99
        model.train(inputs, targets)
        
test_data_file = open("dataset/mnist_test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()

correct = 0
total = 0
for record in test_data_list:
    all_values = record.split(",")
    target = int(all_values[0])
    inputs = (np.asfarray(all_values[1:]))/255.0*0.99+0.01
    pred_idx = np.squeeze(model.query(inputs))
    pred = np.argmax(pred_idx, axis=-1)
    print("pred: %d, target: %d"%(pred, target))
    correct += 1 if pred == target else 0
    total += 1
    
print("准确率为：%f"%(correct/total))