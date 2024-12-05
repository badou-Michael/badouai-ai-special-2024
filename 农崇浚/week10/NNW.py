import numpy as np
import scipy.special

class NNW:
    def __init__(self, inodes, hnodes, onodes, learningrate):
        #初始化网络，设置输入节点，隐藏层的节点，输出节点和学习率
        self.inodes = inodes
        self.hondes = hnodes
        self.onodes = onodes
        self.lr = learningrate

        #初始化权重矩阵
        """
        numpy.random.rand 用于生成指定形状的随机数，这些随机数来自区间[0,1)上均匀分布
        """
        self.wih = np.random.rand(self.hondes,self.inodes) - 0.5#减0.5的目的是希望权重矩阵有正有负
        self.who = np.random.rand(self.onodes,self.hondes) - 0.5

        #激活函数
        self.activation_function = lambda x:scipy.special.expit(x)

    def train(self, input_list, targets_list):
        input = np.array(input_list, ndmin=2).T
        target = np.array(targets_list, ndmin=2).T

        # 计算信号输入信号经过输出层后的信号量
        hidden_input = np.dot(self.wih, input)
        # 计算经过输入层的信号经过激活函数得到的信号量
        hidden_outputs = self.activation_function(hidden_input)
        # 计算输出层接受来自中间层的信号量
        final_outputs = np.dot(self.who, hidden_outputs)
        # 计算输出层经过激活函数的信号量
        final = self.activation_function(final_outputs)

        # 计算损失率
        out_errors = target - final

        hidden_errors = np.dot(self.who.T, out_errors * final * (1 - final))
        # 根据误差计算链路权重的更新量，然后把更新加到原来链路权重上
        self.who += self.lr * np.dot((out_errors * final * (1 - final)),
                                     np.transpose(hidden_outputs))
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                     np.transpose(input))
    def query(self,inputs):
        #计算从输入到隐藏层的信号量
        hidden_inputs = np.dot(self.wih,inputs)

        #计算从隐藏层经过激活函数得到的信号量
        hidden_outputs = self.activation_function(hidden_inputs)

        #计算输出层得到的信号量
        final = np.dot(self.who,hidden_outputs)

        #计算输出层经过激活函数得到的信号量
        final_output = self.activation_function(final)

        return final_output

if __name__ == '__main__':
    #初始化网络
    inode = 784
    hnode = 200
    onode = 10
    lr = 0.3
    n = NNW(inode,hnode,onode,lr)

    #读取数据
    training_data_file = open("dataset/mnist_train.csv",'r')
    training_data_list = training_data_file.readlines()
    training_data_file.close()

    #设定网络的训练次数
    epochs = 80

    for i in range(epochs):
        for j in training_data_list:
            all_values = j.split(',')
            inputs = (np.asfarray(all_values[1:]))/255.0 * 0.99 + 0.01
            #设置图片与数值的对应关系
            targets = np.zeros(onode) + 0.01
            targets[int(all_values[0])] = 0.99
            n.train(inputs, targets)

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
        outputs = n.query(inputs)
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






