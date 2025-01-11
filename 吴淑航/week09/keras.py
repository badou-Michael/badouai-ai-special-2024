[1]
# 加载训练集和测试集
from tensorflow.keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

[2]
# 打印测试中的第一张图片来看看
import matplotlib.pyplot as plt

image = test_images[0]
plt.imshow(image, cmap=plt.cm.binary)
# colormap 颜色映射
# plt.show()

[3]
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()  # 使模型能够进行串联

network.add(layers.Dense(512, activation="relu", input_shape=(28 * 28,)))
# 512个神经元
network.add(layers.Dense(10, activation="softmax"))
# 10个输出再通过softmax

network.compile(optimizer="rmsprop", loss="categorical_crossentropy",
                metrics=["accuracy"])
# 对网络进行编译

[4]
# 对数据进行归一化处理
train_images = train_images.reshape((60000, 28 * 28))  # 将训练数据变成一维的
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
# 对标签做修改
from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

[5]
# 开始训练
network.fit(train_images, train_labels, epochs=4, batch_size=128)

[6]
# 进行测试
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(test_loss)
print('test_acc', test_acc)

[7]
# 推理过程
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
image2 = test_images[1]
plt.imshow(image2, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28 * 28))
res = network.predict(test_images)
# res存储了10000 * 10 的数组
for i in range(res[1].shape[0]):
    if (res[1][i] == 1):  # 循环10次
        print("the number of the picture is : ", i )
        break
