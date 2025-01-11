import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

def Normalization(x):
    return [(float(i)-min(x))/float(max(x)-min(x)) for i in x]

(train_images_all, train_labels_all) = mnist.load_data()[0]
(test_images, test_labels) = mnist.load_data()[1]
num_train_samples = len(train_images_all)
num_train = int(num_train_samples * 0.8)
num_val = num_train_samples - num_train
indices = np.arange(num_train_samples)
np.random.shuffle(indices)
train_images_all = train_images_all[indices]
train_labels_all = train_labels_all[indices]
#训练集&验证集
train_images = train_images_all[:num_train]
train_labels = train_labels_all[:num_train]
val_images = train_images_all[num_train:]
val_labels = train_labels_all[num_train:]

print('train_images.shape = ', train_images.shape)
print('train_labels.shape = ', train_labels.shape)
print('val_images.shape = ', val_images.shape)
print('val_labels.shape = ', val_labels.shape)
print('test_images.shape = ', test_images.shape)
print('test_labels.shape = ', test_labels.shape)
# train_images.shape =  (48000, 28, 28)
# train_labels.shape =  (48000,)
# val_images.shape =  (12000, 28, 28)
# val_labels.shape =  (12000,)
# test_images.shape =  (10000, 28, 28)
# test_labels.shape =  (10000,)

train_images_reshaped = train_images.reshape((num_train, 28 * 28))
train_images_list = train_images_reshaped.tolist()  
normalized_train_images_list = [Normalization(sample) for sample in train_images_list] 
normalized_train_images = np.array(normalized_train_images_list).reshape((num_train, 28*28))  
train_images = normalized_train_images.astype('float32')

val_images_reshaped = val_images.reshape((num_val, 28 * 28))
val_images_list = val_images_reshaped.tolist()  
normalized_val_images_list = [Normalization(sample) for sample in val_images_list] 
normalized_val_images = np.array(normalized_val_images_list).reshape((num_val, 28*28))  
val_images = normalized_val_images.astype('float32')

test_images_reshaped = test_images.reshape((len(test_images), 28 * 28))
test_images_list = test_images_reshaped.tolist()
normalized_test_images_list = [Normalization(sample) for sample in test_images_list]
normalized_test_images = np.array(normalized_test_images_list).reshape((len(test_images), 28*28))
test_images = normalized_test_images.astype('float32')

train_labels = to_categorical(train_labels)
val_labels = to_categorical(val_labels)
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])
network.fit(train_images, train_labels, epochs=5, batch_size=128,
            validation_data=(val_images, val_labels))

# Epoch 1/5
# 375/375 [==============================] - 53s 134ms/step - loss: 0.2909 - accuracy: 0.9158 - val_loss: 0.1502 - val_accuracy: 0.9561
# Epoch 2/5
# 375/375 [==============================] - 57s 152ms/step - loss: 0.1225 - accuracy: 0.9643 - val_loss: 0.0998 - val_accuracy: 0.9697
# Epoch 3/5
# 375/375 [==============================] - 56s 150ms/step - loss: 0.0815 - accuracy: 0.9759 - val_loss: 0.0837 - val_accuracy: 0.9750
# Epoch 4/5
# 375/375 [==============================] - 62s 164ms/step - loss: 0.0591 - accuracy: 0.9821 - val_loss: 0.0738 - val_accuracy: 0.9778
# Epoch 5/5
# 375/375 [==============================] - 58s 155ms/step - loss: 0.0442 - accuracy: 0.9871 - val_loss: 0.0824 - val_accuracy: 0.9747

test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# 313/313 [==============================] - 13s 38ms/step - loss: 0.0775 - accuracy: 0.9747
# 0.07754241675138474
# test_acc 0.9746999740600586


