# 导入所需模块
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
# 将图片展平为一维数组，并进行归一化处理
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# 将标签转换为 One-Hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 创建模型
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))  # 输入层
model.add(layers.Dense(10, activation='softmax'))  # 输出层

# 编译模型
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=128)

# 测试模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")

# 使用模型进行预测
predictions = model.predict(test_images)

# 显示一张测试图片及其预测结果
digit = test_images[0].reshape(28, 28)
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

predicted_label = predictions[0].argmax()
print(f"Predicted label: {predicted_label}")
