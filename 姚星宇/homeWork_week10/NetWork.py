import numpy as np
import scipy.special

class NetWork:
    # 输入层节点数，输出层节点数，隐藏层节点数，学习率
    def __init__(self, inNodes, hideNodes, outNodes, learningRate):
        self.inNodes = inNodes
        self.hideNodes = hideNodes
        self.outNodes = outNodes
        self.learningRate = learningRate
        # 权重矩阵，均值为1，标准差为pow(self.hideNodes, -0.5),矩阵形状为(hideNodes, inNodes)、(outNodes, hideNodes)
        self.wih = (np.random.normal(0.0, pow(self.hideNodes, -0.5), (self.hideNodes, self.inNodes)))
        self.who = (np.random.normal(0.0, pow(self.outNodes, -0.5), (self.outNodes, self.hideNodes)))
        # 激活函数sigmoid
        self.activation_fun = lambda x:scipy.special.expit(x)

    # 训练，更新权重
    def train(self, input_list, targets_list):
        # 数据处理
        inputs = np.array(input_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin=2).T
        # 正向传播
        hideInputs = np.dot(self.wih, inputs)
        hideOutputs = self.activation_fun(hideInputs)
        finalInputs = np.dot(self.who, hideOutputs)
        finalOutputs = self.activation_fun(finalInputs)
        # 计算损失
        output_loss = targets - finalOutputs
        hide_loss = np.dot(self.who.T, output_loss * finalOutputs * (1 - finalOutputs))
        # 反向传播更新权重
        self.who += self.learningRate * np.dot((output_loss * finalOutputs * (1 - finalOutputs)), np.transpose(hideOutputs))
        self.wih += self.learningRate * np.dot((hide_loss * hideOutputs * (1 - hideOutputs)), np.transpose(inputs))

    def run(self, inputs):
        hideInputs = np.dot(self.wih, inputs)
        hideOutputs = self.activation_fun(hideInputs)
        finalInputs = np.dot(self.who, hideOutputs)
        finalOutputs = self.activation_fun(finalInputs)
        print(finalOutputs)
        return finalOutputs
    
if __name__ == "__main__":
    # 定义、训练网络
    inputNodes = 784
    hideNodes = 100
    outputNodes = 10
    learningRate = 0.1
    netWork = NetWork(inputNodes, hideNodes, outputNodes, learningRate)
    # 读取训练数据
    trainDataFile = open("./dataset/mnist_train.csv", 'r')
    trainDataList = trainDataFile.readlines()
    trainDataFile.close()
    # 进行训练
    epochs = 10
    for e in range(epochs):
        for record in trainDataList:
            allValues = record.split(',')
            # 数据预处理，避免输入值为0或1，导致的sigmoid的梯度消失
            inputs = (np.asfarray(allValues[1:])) / 255 * 0.99 + 0.01
            targets = np.zeros(outputNodes) + 0.01
            targets[int(allValues[0])] = 0.99
            netWork.train(inputs, targets)

    # 测试网络
    testDataFile = open("./dataset/mnist_test.csv")
    testDataList = testDataFile.readlines()
    testDataFile.close()
    scores = []
    for record in testDataList:
        allValues = record.split(',')
        curNumber = int(allValues[0])
        print("真实值为:",curNumber)
        inputs = (np.asfarray(allValues[1:])) / 255.0 * 0.99 + 0.01
        outputs = netWork.run(inputs)
        label = np.argmax(outputs)
        print("预测值为：", label)
        if label == curNumber:
            scores.append(1)
        else:
            scores.append(0)
    print(scores)
    # 计算准确率
    scoresArray = np.asarray(scores)
    print("accuracy : ", scoresArray.sum() / scoresArray.size)





