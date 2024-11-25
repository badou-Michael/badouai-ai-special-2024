'''
week09
1.实现标准化
2.使用keras实现简单神经网络
'''
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#1、实现标准化
#用课上的矩阵例子
data = [-10, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11,
     11, 12, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 15, 15, 30]

cs=[]
for i in data:
    c=data.count(i)
    cs.append(c)

standardized_data = zscore(data)
print(standardized_data)

numpy_data = np.array(data).reshape(-1, 1)
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_data = scaler.fit_transform(numpy_data)
print("归一化到 [0, 1]:", normalized_data.flatten())

plt.plot(standardized_data,cs)
plt.plot(normalized_data,cs)
plt.show()

#2、使用keras实现简单神经网络
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

#加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = test_images[0]  # 取出测试集中第 1 张图片
plt.imshow(digit, cmap=plt.cm.binary)
plt.axis('off')  # 隐藏坐标轴（可选）
plt.show()

#按顺序构建全连接神经网络
network = models.Sequential()
network.add(layers.Dense(2056,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#处理输入数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#one-hot编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#输入数据训练
network.fit(train_images, train_labels, epochs=5, batch_size=128)

#打印效果
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

#预测一张新数据
digit = digit.reshape((1, 28 * 28))  # 展平为 1x784
digit = digit.astype('float32') / 255  # 归一化
# 模型预测
predicted_result = network.predict(digit)
predicted_label = np.argmax(predicted_result)  # 取概率最高的类
print(f"预测结果为: {predicted_label}")

