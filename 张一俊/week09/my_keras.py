from tensorflow.keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt

#  1. 加载数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print("\nx_train.shape: \n", x_train.shape)
print("\nx_test.shape: \n", x_test.shape)
# print("\nx_train data: \n", x_train)
print("\ny_train data: \n", y_train)
# print("\nx_test data: \n", x_test)
print("\ny_test data: \n", y_test)

#  2. 数据预处理
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32') / 255

# 转为一维
x_train = x_train.reshape((60000, 28*28))
x_test = x_test.reshape((10000, 28*28))

# 将标签转换为 one-hot 编码
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

#  3.构建模型
model = Sequential()
# 对输入图像的展平有两种方法
# model.add(Flatten(input_shape=(28, 28)))
# model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu', input_shape=(28*28,)))
model.add(Dense(10, activation='softmax'))

# 4.编译模型
# optimizer:权重优化方法， loss：损失函数，accuracy：模型评估时需要计算的指标
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])

# 5.训练模型
model.fit(x_train, y_train, epochs=5, batch_size=32)

# 6.评估模型
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy:{test_acc}")

# 7. 使用训练好的模型进行预测
(x_train, y_train), (x_test, y_test) = mnist.load_data()
digit = x_test[3]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
x_test = x_test.reshape((10000, 28*28))

predictions = model.predict(x_test)

print(predictions.shape)
print(predictions.shape[0])
print(predictions[0])
print(predictions[3])
print(predictions[3].shape)
print(predictions[3].shape[0])


for i in range(predictions[3].shape[0]):
    if predictions[3][i] == 1:
        print("the number for the picture is : ", i)
        break
