import random
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.utils import to_categorical

# [1] 加载MNIST数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print('train_images.shape = ',train_images.shape)
print('tran_labels = ', train_labels)
print('test_images.shape = ', test_images.shape)
print('test_labels = ', test_labels)

# [2] 构建神经网络模型
neural_network = models.Sequential()
neural_network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
neural_network.add(layers.Dense(10, activation='softmax'))

# [3] 编译模型
neural_network.compile(optimizer='rmsprop',
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# [4] 预处理数据 数据归一化处理
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32')/255

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

# [5] 转换标签为one-hot编码 例如 5->[0,0,0,0,0,1,0,0,0,0]
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# [6] 训练模型
neural_network.fit(train_images, train_labels, epochs=3, batch_size=128)

# [7] 评估模型
test_loss, test_acc = neural_network.evaluate(test_images, test_labels, verbose=1)
print(test_loss)
print('test_acc', test_acc)

# [8] 预测测试集   此处仅做演示，实际应用中，预测的数据集为未知的，需要使用模型进行预测
# test_images = test_images.reshape((10000, 28*28))
# test_images = test_images.astype('float32')/255
predict_result = neural_network.predict(test_images)
# print(predict_result)
for m in random.sample(range(10000), 3):
    # print(predict_result[m])
    max_index = np.argmax(predict_result[m])
    print(max_index)

    one_hot = np.zeros_like(predict_result[m])
    one_hot[max_index] = 1
    print(one_hot)

    for n in range(predict_result[m].shape[0]):
        if (predict_result[m][n] == 1):
            print("the number for the picture is : ", n)
            break

    # 输出预测结果
    predict_number = np.argmax(predict_result[m])
    print('识别的数字是：', predict_number)

    # 显示图像
    plt.imshow(test_images[m].reshape(28, 28), cmap='gray')
    plt.title(f"Predicted Number:{test_labels[m]} ")
    plt.show()
