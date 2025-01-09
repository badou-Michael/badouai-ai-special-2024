import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np

# 加载 MNIST 数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
# 归一化：将像素值缩放到 [0, 1] 范围内
x_train = x_train / 255.0
x_test = x_test / 255.0

# 将标签转换为独热编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# 构建神经网络模型
model = models.Sequential()

# 输入层，Flatten 将 28x28 的二维图像展开为 一维向量
model.add(layers.Flatten(input_shape=(28, 28)))

# 隐藏层，包含 128 个神经元，使用 ReLU 激活函数
model.add(layers.Dense(128, activation='relu'))

# 输出层，包含 10 个神经元，对应 10 类（数字 0-9），使用 softmax 激活函数
model.add(layers.Dense(10, activation='softmax'))

# 编译模型
model.compile(optimizer='adam',  # 使用 Adam 优化器
              loss='categorical_crossentropy',  # 使用交叉熵损失函数
              metrics=['accuracy'])  # 监控准确率

# 训练模型
history = model.fit(x_train, y_train, epochs=5, batch_size=64, validation_split=0.2)

#  评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')

#  使用训练后的模型进行手写数字识别
# 随机选择一张图片
random_idx = np.random.randint(0, len(x_test))  # 随机选择索引
random_image = x_test[random_idx]  # 获取图像数据
random_label = y_test[random_idx]  # 获取标签

# 显示随机选择的图像
plt.imshow(random_image, cmap='gray')
plt.title(f"True Label: {np.argmax(random_label)}")
plt.show()

# 进行预测
predicted_label = model.predict(np.expand_dims(random_image, axis=0))  # 预测，模型输入需要是批次形式
predicted_digit = np.argmax(predicted_label)  # 获取预测结果（数字）

print(f"Predicted Label: {predicted_digit}")
