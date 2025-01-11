from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape)
print(train_labels)
print(test_images.shape)
print(test_labels)

# 构建模型
network = models.Sequential()
# 构建网络层 输入层 28*28  隐藏层 512个神经元 relu激活函数
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# 隐藏层 128 relu激活函数
network.add(layers.Dense(128, activation='relu'))
# 输出层 10个神经元 softmax激活函数
network.add(layers.Dense(10, activation='softmax'))
# 编译模型
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 准备数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical

print("before change:", test_labels[0]) # 值为7
print("before train_labels change:", train_labels[0]) # 值为5
train_labels =to_categorical(train_labels) # 转换为one-hot编码
test_labels = to_categorical(test_labels)#   转换为one-hot编码
print("after change:", test_labels[0]) # 值为[0. 0. 0. 0. 0. 0. 1. 0. 0. 0.] 0-9对应10个神经元 7对应第8个神经元
print("after train_labels change:", train_labels[0])# 值为[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]

network.fit(train_images, train_labels, epochs=5, batch_size=128)
test_loss, test_acc = network.evaluate(test_images, test_labels)
print('test_loss:', test_loss)
print('test_acc:', test_acc)
# 预测
result = network.predict(test_images)
for i in range(result[1].shape[0]):
    if result[1][i] == 1:
        print(i)
        break
