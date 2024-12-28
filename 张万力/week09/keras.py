from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist

# 1. 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 2. 数据预处理
# 归一化
x_train = x_train / 255.0
x_test = x_test / 255.0

# 3. 构建模型
model = models.Sequential([
    layers. Flatten(input_shape=(28, 28)),  # 将28x28的图像展平为784维向量
    layers. Dense(128, activation='relu'),   # 第一个隐藏层，128个神经元
    layers. Dense(64, activation='relu'),    # 第二个隐藏层，64个神经元
    layers.Dense(10, activation='softmax')  # 输出层，10个类别
])

# 4. 编译模型
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. 训练模型
model.fit(x_train, y_train,
          epochs=5,
          batch_size=128,
          validation_data=(x_test, y_test))

# 6. 评估模型
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"测试准确率: {test_accuracy}")
