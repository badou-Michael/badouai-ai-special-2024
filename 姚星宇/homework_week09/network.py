from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[0]
import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
# 构建模型
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))

# 编译模型
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# 准备训练数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
train_labels = to_categorical(train_labels)

# 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 准备测试数据
test_images = test_images.reshape((10000, 28 * 28))  # 重塑测试图像
test_images = test_images.astype('float32') / 255  # 归一化测试图像
test_labels = to_categorical(test_labels)

# 评估模型
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# 预测测试数据
res = network.predict(test_images)
predicted_digit = np.argmax(res[0])
print("Predicted digit for the first test image is:", predicted_digit)

# 输出预测结果
for i in range(res[0].shape[0]): 
    if res[0][i] == 1:
        print("Number is:", i)
        break
