# 导入必要的模块
import matplotlib.pyplot as plt
from tensorflow.keras import models
from tensorflow.keras import layers

# 加载数据mnist
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)


# 搭建神经网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))   # 输入数据是28*28的二维数组
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(64, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

# 优化器：adam，rmsprop；交叉熵损失函数categorical_crossentropy；
network.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['accuracy'])
# 数据归一化
train_images = train_images.reshape((60000, 28*28))  # 变成28*28的一维数组
train_images = train_images.astype('float32') / 255     # 归一化

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255
# 输出0-9
from tensorflow.keras.utils import to_categorical
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size = 128)

# 测试数据效果
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 新的数据识别测试
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)

for i in range(res[4].shape[0]):
    if (res[4][i] == 1):
        print("the number for the picture is : ", i)
        break