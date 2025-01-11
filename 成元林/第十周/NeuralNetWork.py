import numpy as np
import scipy.special
import pandas as pd


class NeuralNetWork:
    def __init__(self, inputnodes, hidenodes, outputnodes, inputb, outputb, learnrate):
        # 初始化输入节点
        self.innodes = inputnodes
        # 隐藏层节点
        self.hnodes = hidenodes
        # 输出层节点
        self.outnodes = outputnodes
        self.inb = inputb
        self.outb = outputb
        # 初始化权重，shape应该为（inputnodes，hidenodes），[[w1,w2],[w3,w4]]
        # 生成一个形状为 (m, n) 的二维数组，包含随机浮点数,值为0.0-1.0之间，权重有可能是负值，-0.5是让权重-0.5-0.5之间
        # 输入层到隐藏层权重初始化
        self.inw = np.random.rand(self.innodes, self.hnodes) - 0.5
        # 隐藏层到输出层权重
        self.outw = np.random.rand(self.hnodes, self.outnodes) - 0.5
        print("self.inw.shape:",self.inw.shape)
        print("self.outw.shape:",self.outw.shape)
        self.activeFunction = lambda x: scipy.special.expit(x)
        self.learnrate = learnrate
        pass

    def train(self, inputResults, targetinputs):
        """
        训练过程，通过误差更新权重
        @param inputResults: 通过一代训练得到的结果
        @param targetinputs: 实际结果
        @return:
        """
        # 转为二维数组矩阵,形状是（1，784）self.inw形状为(784,200)
        inputResults = np.array(inputResults, ndmin=2)
        print(inputResults.shape)
        targetinputs = np.array(targetinputs, ndmin=2)

        # np.dot对于二维数组是矩阵乘法,这里公式是y = x*w+b
        hideInputs = np.matmul(inputResults, self.inw) + self.inb
        # hideInputs经过激活函数则变成hideOutputs
        hideOutputs = self.activeFunction(hideInputs)
        # 隐藏层输出作为输出层输入参数，根据wx+b
        final_inputs = np.matmul(hideOutputs, self.outw) + self.outb
        finalOutputs = self.activeFunction(final_inputs)
        # 误差值
        outputError = targetinputs - finalOutputs


        print(finalOutputs.shape)
        print("hideOutputs.shape:",hideOutputs.shape)
        print(self.outw.shape)
        # 根据损失函数公式更新权值w,wnew = w-learnrate*w偏导  ,偏导公式：pd = -(tk-ok)*ao1*(1-ao1)*ah1
        # self.outw = self.outw - (hideOutputs.T)*(-(outputError) * finalOutputs * (1 - finalOutputs) )
        hidden_errors = np.dot(outputError * finalOutputs * (1 - finalOutputs),self.outw.T)
        #权重更新
        self.outw = self.outw + self.learnrate * np.dot(hideOutputs.T, outputError * finalOutputs * (1 - finalOutputs))
        self.inw = self.inw + self.learnrate * np.dot(inputResults.T,hidden_errors*hideOutputs*(1-hideOutputs))
    def query(self, inputs,inw,outw):
        """
         y = wx+b 这里w*x顺序存在问题，应该输入乘权重，所以这里应该是x*w
        @return:
        """
        # np.dot对于二维数组是矩阵乘法,这里公式是y = x*w+b
        print(self.inw.shape)
        hideInputs = np.matmul(inputs, inw) + self.inb
        # hideInputs经过激活函数则变成hideOutputs
        hideOutputs = self.activeFunction(hideInputs)
        # 隐藏层输出作为输出层输入参数，根据wx+b
        outinputs = np.matmul(hideOutputs, outw) + self.outb
        finalOutputs =  self.activeFunction(outinputs)
        return finalOutputs


if __name__ == '__main__':
    inputNodes = 784
    hiddenNodes = 100
    outputNodes = 10
    inputb = 0.3
    outputb = 0.5
    learnrate = 0.1
    n = NeuralNetWork(inputNodes, hiddenNodes, outputNodes, inputb, outputb, learnrate)
    # 读取训练集数据
    traindata = pd.read_csv("./dataset/mnist_train.csv")
    # dataframe转为ndarray
    traindatalist = traindata.values
    # 遍历数据集每行数据
    epoch = 15
    for j in range(epoch):
        for i in range(traindatalist.shape[0]):
            record = traindatalist[i]
            # 每一条记录的第一列是正确答案
            #这个是保证每个数据都是再0.01-1之间
            inputs = record[1:] / 255.0 * 0.99 + 0.01
            targets = np.zeros(outputNodes) + 0.01
            targets[int(record[0])] = 1
            n.train(inputs,targets)

    testdata = pd.read_csv("./dataset/mnist_test.csv")
    testdataList = testdata.values
    scores = []
    for i in range(testdataList.shape[0]):
        testrecord = testdataList[i]
        print("该图片对应的数字为:", testrecord[0])
        inputs = testrecord[1:] / 255.0 * 0.99 + 0.01
        outputs = n.query(inputs)
        # print("test-outputs:",outputs)
        label = np.argmax(outputs)
        print("网络认为图片的数字是：", label)
        if label == int(testrecord[0]):
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    scores_array = np.asarray(scores)
    print("perfermance = ", scores_array.sum() / scores_array.size)