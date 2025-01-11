import numpy as np
import matplotlib.pyplot as plt
import statistics
from tensorflow.keras.datasets import mnist
from keras import models
from keras import layers
from tensorflow.keras.utils import to_categorical


# 1. 实现标准化
def normalization(x):
    return [float((i - min(x))) / float(max(x) - min(x)) for i in x]

def normalization_1(x):
    return [float((i - np.mean(x))) / float(max(x) - min(x)) for i in x]

def z_score(x):
    m = np.mean(x)
    n = statistics.stdev(x)
    return [(i-m)/n for i in x]

Y_value = [1,2,3,4,5,6,7,8]
A = [3,6,7,-4,-7,8,5,10]
B = normalization(A)
C = normalization_1(A)
D = z_score(A)
plt.plot(A,Y_value)
plt.plot(B,Y_value)
plt.plot(C,Y_value)
plt.plot(D,Y_value)
plt.show()




# 2. 使用keras实现简单神经网络
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()  # load mnist data

# model building
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(256, activation='relu'))
network.add(layers.Dense(128, activation='relu'))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

# images & labels pre-processing
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# train
network.fit(train_images, train_labels, epochs=10, batch_size=128)

# test
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print('test_loss', test_loss)
print('test_acc', test_acc)

# predict
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28 * 28))
res = network.predict(test_images)
for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break
