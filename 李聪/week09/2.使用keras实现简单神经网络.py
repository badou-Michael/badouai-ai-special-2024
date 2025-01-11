from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255  # 归一化
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255  # 归一化

train_labels = to_categorical(train_labels)  # 转换为one-hot编码
test_labels = to_categorical(test_labels)

# 构建神经网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))  # 输入层+隐藏层
network.add(layers.Dense(10, activation='softmax'))  # 输出层

# 编译模型
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 测试模型
test_loss, test_acc = network.evaluate(test_images, test_labels)
print("测试集损失：", test_loss)
print("测试集准确率：", test_acc)
