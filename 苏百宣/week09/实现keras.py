# 实现keras author：苏百宣
# 1. 加载 MNIST 数据集
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 查看数据的基本信息
print('train_images.shape = ', train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)

# 2. 数据预处理
# 将二维图像数据展平成一维，并归一化到 [0, 1]
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# 将标签转为 One-Hot 编码
print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change:", test_labels[0])

# 3. 构建神经网络
from tensorflow.keras import models, layers

# 搭建神经网络
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 编译模型
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# 4. 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 5. 测试模型
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc)

# 6. 可视化数据（放到最后）
# 查看测试集中第一张图片及其预测结果
digit = test_images[0].reshape(28, 28)  # 恢复为二维图片
plt.imshow(digit, cmap=plt.cm.binary)
plt.title("Test Image")
plt.show()

# 让模型对测试集中第一张图片进行预测
predictions = network.predict(test_images)
predicted_label = predictions[0].argmax()  # 取最大概率对应的类别
print(f"The model predicts this image as: {predicted_label}")
