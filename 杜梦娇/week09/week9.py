import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 标准化数据
# 随机生成一组随机数
test_data = np.random.randint(0,1000, [1,1000], dtype=int)

#(x−x_min)/(x_max−x_min)
def myNormalization1(data):
    return (data - np.min(data)) / (np.max(data)-np.min(data))
#(x−x_mean)/(x_max−x_min)
def myNormalization2(data):
    return (data - np.mean(data)) / (np.max(data)-np.min(data))
#z-score
def myNormalization3(data):
    return (data - np.mean(data)) / np.std(data)

data_normalized1 = myNormalization1(test_data)
data_normalized2 = myNormalization2(test_data)
data_normalized3 = myNormalization3(test_data)
print("原始数据:", test_data)
print("(x−x_min)/(x_max−x_min):", data_normalized1)
print("(x−x_mean)/(x_max−x_min):", data_normalized2)
print("Z-SCORE:", data_normalized3)

#====================================================================
# 基于Keras和手写数据集建立一个简单神经网络进行分类
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels: ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels: ', test_labels)

# 数据预处理
train_images = train_images.reshape(train_images.shape[0], 28*28).astype('float32') / 255
test_images = test_images.reshape(test_images.shape[0], 28*28).astype('float32') / 255
print('train_images.shape = ',train_images.shape)
print('test_images.shape = ', test_images.shape)
# 标签one-hot处理
train_labels = to_categorical(train_labels, 10)
test_labels = to_categorical(test_labels, 10)
print('tran_labels = ', train_labels)
print('test_labels', test_labels)

# 构建一个Sequential模型
model = Sequential()

# 添加第一个全连接层，784个输入特征，512个输出特征
model.add(Dense(512, input_shape=(28*28,), activation='relu'))

# 添加输出层，10个输出特征对应10个类别
model.add(Dense(10, activation='softmax'))

# 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, batch_size=128, epochs=10)

# 评估模型
score = model.evaluate(test_images, test_labels, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# 推理新的图片
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
data = test_images[100]
plt.imshow(data, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = model.predict(test_images)

for i in range(res[100].shape[0]):
    if (res[100][i] == 1):
        print("the number is : ", i)
        break
