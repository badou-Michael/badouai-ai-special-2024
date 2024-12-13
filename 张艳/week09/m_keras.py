[1]
from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('-------------------------------------------------- mnist.load_data()')
print('train_images.shape = ',train_images.shape)
print('train_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels', test_labels)
print('-------------------------------------------------- ')

[2]
import matplotlib.pyplot as plt
digit = test_images[0]
plt.imshow(digit, cmap=plt.cm.binary)
# plt.show()

[3]
from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential() # 创建一个顺序模型
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,))) # 添加第一层（全连接层），有64个神经元的全连接层（Dense layer），输入维度为28*28
network.add(layers.Dense(10, activation='softmax')) # 添加第二层（全连接层）

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy']) # 编译模型，指定优化器 optimizer、损失函数 loss、评估指标 metrics

[4]
train_images = train_images.reshape((60000, 28*28)) # 原本(60000,28,28)
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28)) # 原本(10000,28,28)
test_images = test_images.astype('float32') / 255


from tensorflow.keras.utils import to_categorical

print('-------------------------------------------------- to_categorical()')
print("before change:" ,test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
print("after change: ", test_labels[0])
print('-------------------------------------------------- ')

[5]
print('-------------------------------------------------- network.fit()')
network.fit(train_images, train_labels, epochs = 5, batch_size = 128)
print('-------------------------------------------------- ')

[6]
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('-------------------------------------------------- network.evaluate()')
print(test_loss)
print('test_acc', test_acc)
print('-------------------------------------------------- ')

[7]
print('-------------------------------------------------- network.predict')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
print('res.shape',res.shape)

# 查看第二张的预测结果
for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
print('-------------------------------------------------- ')
