# coding: utf-8
# Author: ChungRae
# File  : ann_impl.py
# Time  : 2024/11/18 星期一 16:52
# Desc  : 手动搭建ann

from math import exp
import numpy as np
from tensorflow.keras import layers, models, datasets
from tensorflow.keras.utils import to_categorical


def sigmoid(x):
    """sigmoid 激活函数"""
    return 1 / (1 + exp(-x))

def propagation():
    """
    正向传播
    """

    # 预期输出
    eo1, eo2 = 0.01, 0.99

    # 输入层->隐藏层
    i1, i2 = 0.05, 0.1
    # 权重 i1->h1 i2->h1 i1->h2 i2->h2
    iw11, iw21, iw12, iw22 = 0.15, 0.2, 0.25, 0.3
    # 输入层的偏置量
    offset1 = 0.35
    # 隐藏层输入 h = sigmoid(WX+b)
    h1 = sigmoid(i1 * iw11 + i2 * iw21 + offset1)
    h2 = sigmoid(i1 * iw12 + i2 * iw22 + offset1)

    # 隐藏层->输出层
    # 隐藏层权重 h1->o1 h2->o1 h1->o2 h2->o2
    hw11, hw21, hw12, hw22 = 0.4, 0.45, 0.5, 0.55


    # 隐藏层的偏置量
    offset2 = 0.6
    # 隐藏层输出 o = sigmoid(WX + b)
    o1 = sigmoid(h1 * hw11 + h2 * hw21 + offset2)
    o2 = sigmoid(h1 * hw12 + h2 * hw22 + offset2)

    # o1 o2和预期的差值用mae
    diff_o1 = (eo1 - o1) ** 2 / 2
    diff_o2 = (eo2 - o2) ** 2 / 2

    # 总损失
    loss = diff_o1 + diff_o2


    def back_propagation(t, o ,h, w, learning_rate=0.5):
        # 反向传播
        # 隐藏层->输出层的权重 delta_e/delta_w = delta_e/delta_o * delta_o/delta_z * delta_z/delta_w

        # e = 1/2 * (eo1-o1)**2 + 1/2 *(eo2-o2)**2
        # delta_eo = -(eo1 - o1)
        # delta_eo =  -(eo1 - o1)

        # o = 1/(1+e^-z)
        # delta_oz = z*(1-z)
        # delta_oz = h1 * (1-h1)

        # z =  h1 * hw11 + h2 * hw21 + offset2
        # delta_zw = h1

        delta_zw = h1


        #delta_ew = delta_eo * delta_oz * delta_zw
        new_w  = w - learning_rate * (-(t - o) * h*(1-h) * h)

        # 输入层->隐藏层的权重
        # delta_e/delta_w = delta_e/delta_h * delta_h/delta_z * delta_z/delta_w

        return new_w


    new_hw11 = back_propagation(eo1, o1, h1, hw11)
    new_hw21 = back_propagation(eo1, o1, h1, hw21)
    new_hw12 = back_propagation(eo2, o2, h2, hw12)
    new_hw22 = back_propagation(eo2, o2, h2, hw22)

    print(new_hw11, new_hw21, new_hw12, new_hw22)

def max_min_normalization(data: np.ndarray):
    """
    最大最小归一化
    @param data: 数据
    @return:
    """
    max_val = np.max(data)
    min_val = np.min(data)
    return (data - min_val) / (max_val - min_val)

def z_score_normalization(data: np.ndarray):
    """
    均值方差归一化
    @param data:
    @return:
    """
    mean_val = np.mean(data)
    std_val = np.std(data)
    return (data - mean_val) / std_val

def construct_ann_by_keras():
    """
    使用keras构建ann
    @return:
    """
    # 创建模型
  
    model = models.Sequential()

    # 构建输入层，激活函数是relu
 
    model.add(layers.Dense(512, activation='relu', input_shape=(28*28, )))

    # 构建输出层，激活函数是softmax
    model.add(layers.Dense(10, activation='softmax'))

    # 模型编译, 损失函数用交叉熵
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'])

    # 加载数据
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

    # 转换为28*28, 归一化
    x_train = x_train.reshape((60000, 28*28)).astype('float32') / 255
    x_test = x_test.reshape((10000, 28*28)).astype('float32') / 255

    # 按分类标准转化为二进制矩阵，比如是3，只有第3个为1，其他都是0
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # 使用训练集训练模型
    model.fit(x_train, y_train, epochs=10, batch_size=128)

    # 使用测试集测试模型
    loss, acc = model.evaluate(x_test, y_test, verbose=1)

    print(f"loss: {loss}, acc: {acc}")

    # 数据验证
    _, (x_test, _) = datasets.mnist.load_data()
    x_test = x_test.reshape((10000, 28*28))
    res = model.predict(x_test)
    print(res)




if __name__ == '__main__':
    # propagation()
    # print(max_min_normalization(np.arange(4)))
    # print(z_score_normalization(np.arange(5)))
    construct_ann_by_keras()
