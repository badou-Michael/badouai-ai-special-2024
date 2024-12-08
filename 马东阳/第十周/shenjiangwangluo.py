# 神将网络
import numpy as np
import scipy.special

class NeuralNetWork:
    def __init__(self, input, hidden, output, learningrate):
        self.input = input
        self.hidden = hidden
        self.output = output

        self.lr = learningrate
        self.wih = (np.random.normal(0.0, pow(self.hidden, -0.5),(self.hidden, self.input)))
        self.who = (np.random.normal(0.0, pow(self.output, -0.5), (self.output, self.hidden)))

        self.activation_function = lambda x:scipy.special.expit(x)
        pass

    def train(self, input_list, target_list):
        input = np.array(input_list, ndmin = 2).T
        target = np.array(target_list, ndmin = 2).T

        hidden_inputs = np.dot(self.wih, input)
        hidden_outputs = self.activation_function(hidden_inputs)
        #
        final_inputs = np.dot(self.who, hidden_outputs)
        #
        final_outputs = self.activation_function(final_inputs)

        #
        output_errors = target - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors * final_outputs * (1 - final_outputs))
        #
        self.who += self.lr * np.dot((output_errors * final_outputs * (1 - final_outputs)),
                                        np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                        np.transpose(input))
        pass

    # tuili
    def query(self, inputs):
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

input_num = 784
hidden_num = 200
output_num = 10
learning_rate = 0.05

model = NeuralNetWork(input_num, hidden_num, output_num, learning_rate)

#读入训练数据
#open函数里的路径根据数据存储的路径来设定
training_data_file = open("dataset/mnist_train.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

#加入epocs,设定网络的训练循环次数
epochs = 5
for e in range(epochs):
    #把数据依靠','区分，并分别读入
    for record in training_data_list:
        all_values = record.split(',')
        inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
        #设置图片与数值的对应关系
        targets = np.zeros(output_num) + 0.01
        targets[int(all_values[0])] = 0.99
        model.train(inputs, targets)

test_data_file = open("dataset/mnist_test.csv")
test_data_list = test_data_file.readlines()
test_data_file.close()
scores = []
for record in test_data_list:
    all_values = record.split(',')
    correct_number = int(all_values[0])
    print("该图片对应的数字为:",correct_number)
    #预处理数字图片
    inputs = (np.asfarray(all_values[1:])) / 255.0 * 0.99 + 0.01
    #让网络判断图片对应的数字
    outputs = model.query(inputs)
    #找到数值最大的神经元对应的编号
    label = np.argmax(outputs)
    print("网络认为图片的数字是：", label)
    if label == correct_number:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

#计算图片判断的成功率
scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
