# 准备数据集-手写数字图片
from tensorflow.keras.datasets import mnist

(train_img, train_labels), (test_img, test_labels) = mnist.load_data()
# 打印数据集出来看看
print(train_img.shape)
print(test_img.shape)
import matplotlib.pyplot as plt

img_show = train_img[0]
plt.imshow(img_show, cmap="gray")
plt.show()
# 构建深度神经网络
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation="softmax"))
network.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
# 处理数据，归一化，扁平化,标签要one hot
train_img = train_img.reshape((60000, 28 * 28))
test_img = test_img.reshape((10000, 28 * 28))
train_img = train_img.astype('float32') / 255
test_img = test_img.astype('float32') / 255
from tensorflow.keras.utils import to_categorical

print("before change:", test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])

# 开始训练
network.fit(train_img, train_labels, batch_size=128, epochs=5)
# 验证集
loss, acc = network.evaluate(test_img, test_labels, verbose=1)
print("test loss", loss)
print("正确率", acc)
# 推理
(train_img, train_labels), (test_img, test_labels) = mnist.load_data()

digit = test_img[0]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_img = test_img.reshape((10000, 28 * 28))
predict = network.predict(test_img)
print("predict",predict.shape)
for i in range(10):
    if predict[0][i] == 1:
        print('结果是' , i)
        break
