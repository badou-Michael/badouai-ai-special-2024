import tensorflow
from keras.datasets import mnist
from keras import models
from keras import layers
from keras.utils import to_categorical
from matplotlib import pyplot as plt


(train_images, train_labels),(test_images, test_labels)=mnist.load_data()

net = models.Sequential()
net.add(layers.Dense(512, activation = 'tanh', input_shape=(28*28,)))
net.add(layers.Dense(10, activation = 'softmax'))
net.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


net.fit(train_images, train_labels, epoch=10, batch_size = 128)

loss, acc = net.evaluate(test_images, test_labels, verbose = 1)
print(loss)
print('accuracy=', acc)


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[1]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000, 28*28))
res = net.predict(test_images)

for i in range(res[1].shape[0]):
    if (res[1][i] == 1):
        print("the number for the picture is : ", i)
        break