import numpy as np
import scipy.special


class Net:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, lr):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.lr = lr

        self.w_i_h = np.random.normal(
            0, pow(self.hidden_nodes, -0.5), (self.hidden_nodes, self.input_nodes)
        )
        self.w_h_o = np.random.normal(
            0, pow(self.hidden_nodes, -0.5), (self.output_nodes, self.hidden_nodes)
        )

        self.actfunction = lambda x: scipy.special.expit(x)

        pass

    def train(self, input, target):
        input = np.asarray(input).reshape(-1, 1)
        target = np.asarray(target).reshape(-1, 1)

        hidden_input = np.dot(self.w_i_h, input)
        hidden_output = self.actfunction(hidden_input)

        fin_output = self.actfunction(np.dot(self.w_h_o, hidden_output))

        # 计算误差
        output_error = target - fin_output
        hidden_error = np.dot(
            self.w_h_o.T, output_error * fin_output * (1 - fin_output)
        )

        # 更新权重

        self.w_h_o += self.lr * np.dot(
            (output_error * fin_output * (1 - fin_output)), hidden_output.T
        )
        self.w_i_h += self.lr * np.dot(
            (hidden_error * hidden_output * (1 - hidden_output)), input.T
        )

        pass

    def query(self, input):
        hidden_output = self.actfunction(np.dot(self.w_i_h, input))
        fin_output = self.actfunction(np.dot(self.w_h_o, hidden_output))
        print(fin_output)
        return fin_output

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
lr = 0.1
epoch = 10

net = Net(input_nodes, hidden_nodes, output_nodes, lr)

data_train = open("week10\practice\dataset\mnist_train.csv")
data_train_list = data_train.readlines()
data_train.close()

for e in range(epoch):
    for data in data_train_list:
        all_data = data.split(",")
        input = ((np.asarray(all_data[1:])).astype(float)) / 255.0 * 0.99 + 0.01
        target = np.zeros(output_nodes) + 0.01
        target[int(all_data[0])] = 0.99
        net.train(input, target)


data_test = open("week10\practice\dataset\mnist_test.csv")
data_test_list = data_test.readlines()
data_test.close()

scores = []

for data in data_test_list:
    all_data = data.split(",")
    correct_num = int(all_data[0])
    print("该图片对应的数字为:", correct_num)
    input = ((np.asarray(all_data[1:])).astype(float)) / 255.0 * 0.99 + 0.01

    output = net.query(input)
    label = np.argmax(output)
    print("该图片对应的数字为:", label)
    if label == correct_num:
        scores.append(1)
    else:
        scores.append(0)
print(scores)

scores_array = np.asarray(scores)
print("perfermance = ", scores_array.sum() / scores_array.size)
