from tensorflow.keras.datasets import mnist
import cv2
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical


def loadData():
    """
    从keras加载数据
    @return:
    """
    (trainData, trainLabels), (testData, testLabels) = mnist.load_data()
    return (trainData, trainLabels), (testData, testLabels)


def handleData(data, dataSize, imagesH, imageW):
    """
    数据处理，先数据进行标准化/归一化
    @param data: 源数据
    @param dataSize: 数据量
    @param imagesH:图高度
    @param imageW:图宽度
    @return:
    """
    data1 = data.reshape(dataSize, imagesH * imageW)
    data1 = data1.astype('float32') / 255

    return data1,


def one_hot(trainLabels, testLabels):
    # one-hot ,将数字的手写图案，变成一个对应的标签，
    print("before change:", trainLabels[0])
    trainLabels = to_categorical(trainLabels)
    testLabels = to_categorical(testLabels)
    print("after change: ", testLabels[0])
    return trainLabels, testLabels


def setNetwork():
    network = models.Sequential()
    # 设置隐藏层，激活函数用relu
    # 512代表层
    # input_shape(28*28,):表示当前处理层接收的数据格式必须是长和宽都是28的二维数组,
    # 后面的“,“表示数组里面的每一个元素到底包含多少个数字都没有关系.
    network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
    # 输出层,0-9 10个数字，输出层设置10层
    network.add(layers.Dense(10, activation="softmax"))
    return network


def trainNetWork(network, trainData, trainLabels, epochs=5, batchSize=128):
    """
    训练神经网络
    @param network:
    @return:
    """
    # 编译神经网络
    # ‌optimizer‌：优化器，用于定义模型在训练过程中如何更新权重。可以是现有优化器的字符串标识符，如rmsprop或adagrad，也可以是Optimizer类的实例‌
    # loss:损失函数定义了模型优化的目标，选择合适的损失函数可以显著提高模型的训练效果和泛化能力,categorical_crossentropy交叉熵
    # ‌metrics‌：评价指标用于监控训练过程中的性能变化，帮助调整模型参数和防止过拟合
    network.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=['accuracy'])
    network.fit(trainData,trainLabels,epochs=epochs,batch_size=batchSize)
    return network

if __name__ == '__main__':
    (trainData, trainLabels), (testData, testLabels) = loadData()
    print('train_images.shape = ', trainData.shape)
    print('tran_labels = ', trainLabels)
    print('test_images.shape = ', testData.shape)
    print('test_labels', testLabels)
    # #处理数据，归一化，one-hot
    trainData = handleData(trainData,trainData.shape[0],28,28)
    testData = handleData(testData,testData.shape[0],28,28)
    trainLabels,testLabels = one_hot(trainLabels,testLabels)
    # #构建神经网络
    network = setNetwork()
    # #训练神经网络
    network = trainNetWork(network,trainData,trainLabels,epochs=5,batchSize=128)
    # #测试神经网络,返回损失误差，正确率
    # #verbose是否打印日志
    testloss,testacc = network.evaluate(testData,testLabels,verbose=1)
    print("test_loss:",testloss)
    print('test_acc:', testacc)
    # #现场数据识别效果
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    test_images1 = test_images[4].reshape((1, 28 * 28))
    res = network.predict(test_images1)
    print("res:",res)
    for index,i in enumerate(res[0]):
        print(index,i)
        if i==1:
            print("识别的结果是数字：",index)
            break





