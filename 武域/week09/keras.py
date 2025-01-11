import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() # import training data
digit = test_images[0] # display first test image
plt.imshow(digit)
plt.show()

# set up network
from tensorflow.keras import models
from tensorflow.keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# re-shapping
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels) # [0] to 1,0,0,0,0,0,0,0,0
test_labels = to_categorical(test_labels)

network.fit(train_images, train_labels, epochs=5, batch_size = 128) # start trainning

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1) # get result
print(test_loss) 
print('test_acc', test_acc)

test_image = np.expand_dims(test_images[1], axis=0)
res = network.predict(test_image)
predicted_label = np.argmax(res)
print("The number for the picture is:", predicted_label)
