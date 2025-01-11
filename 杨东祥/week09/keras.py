import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

# 1. 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2. 可视化其中一张测试图像
plt.imshow(test_images[2], cmap=plt.cm.binary)
plt.show()

# 3. 数据预处理 - 展平图像并进行归一化
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255

# 4. 标签转换为 One-Hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 5. 构建神经网络模型
model = models.Sequential()
model.add(layers.Dense(128, activation='relu', input_shape=(28 * 28,)))
model.add(layers.Dense(10, activation='softmax'))

# 6. 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. 训练模型
model.fit(train_images, train_labels, epochs=3, batch_size=64)

# 8. 在测试集上评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("Test accuracy:", test_acc)

# 9. 对测试集中的一张图片进行预测
digit = test_images[2].reshape(1, 28*28)  # 重新调整图像形状为 (1, 784)

prediction = model.predict(digit)
predicted_label = np.argmax(prediction)
print(f"The predicted label is: {predicted_label}")
