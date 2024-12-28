from tensorflow.keras.datasets import mnist
# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print("train_images.shape: ", train_images.shape)
print("train_labels.shape: ", train_labels.shape)
print("test_images.shape: ", test_images.shape)
print("test_labels.shape: ", test_labels.shape)

from tensorflow.keras.utils import to_categorical
# 输入形状改变(60000,28,28)->(60000,28*28)
train_images = train_images.reshape((len(train_images), -1))
test_images = test_images.reshape((len(test_images), -1))
# 输入做归一化：0~255 -> 0.0~1.0
train_images = train_images.astype("float32")/255
test_images = test_images.astype("float32")/255
# label改为one-hot编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print(train_labels[0])

from tensorflow.keras import models
from tensorflow.keras import layers
net_work = models.Sequential()
# 添加全连接层+relu激活层
net_work.add(layers.Dense(256, activation="relu"))
# 添加全连接层+softmax层
net_work.add(layers.Dense(10, activation="softmax"))
# 设置优化器，损失函数，评估指标
net_work.compile(optimizer="rmsprop",
                 loss="categorical_crossentropy",
                 metrics="accuracy")

# 训练
net_work.fit(train_images, train_labels, batch_size=256, epochs=5, verbose=1)

# 测试
output = net_work.evaluate(test_images, test_labels, verbose=1)
print("loss: %f, accuracy: %f"%(output[0], output[1]))

_,(test_images, _) = mnist.load_data()
import matplotlib.pyplot as plt
import numpy as np
plt.imshow(test_images[100], cmap=plt.cm.binary)
plt.show()
# 调整test_images形状
test_images = test_images.reshape(test_images.shape[0],-1)
test_images = test_images.astype("float")/255
# 预测
res = net_work.predict(test_images)

print("predict number is ", np.argmax(res[100]))