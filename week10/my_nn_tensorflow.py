import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# 加载数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train / 255.0
x_test = x_test / 255.0

# 创建模型
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # 输入层：将28x28图像展平
    layers.Dense(128, activation='relu'),  # 隐藏层：128个神经元，ReLU激活
    layers.Dropout(0.2),                   # Dropout层
    layers.Dense(10, activation='softmax') # 输出层：10个神经元，softmax激活（用于多分类）
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(x_train, y_train, epochs=5)

# 评估模型
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# 使用训练好的模型进行预测
predictions = model.predict(x_test)

# 显示一个预测结果
plt.imshow(x_test[0], cmap=plt.cm.binary)
plt.title(f"Predicted: {np.argmax(predictions[0])}, Actual: {y_test[0]}")
plt.show()
