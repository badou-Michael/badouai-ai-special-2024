import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras  import layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import cv2
import numpy as np

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

# print('train_images.shape', train_images.shape)
# print('train_labels', train_labels)
# print('test_images.shape', test_images.shape)
# print('test_labels', test_labels)

# 用于测试的第一张图片打印出来看看
# digit = test_images[0]
# plt.imshow(digit,cmap = plt.cm.binary)
# plt.show()

# # 使用tensorflow.Keras搭建一个有效识别图案的神经网络
network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# loaded_model = load_model("number_model")

# 在把数据输入到网络模型之前，把数据做归一化处理
train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float')/255
test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float')/255

#把图片对对应的标记也做一个更改
# print('before change:',test_labels[0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
# print('after change:',test_labels[0])


# 把数据输入到网络进行训练
network.fit(train_images,train_labels,epochs=1,batch_size=128)

# 保存模型
# network.save('number_model')

# 评估模型
# test_loss,test_acc = network.evaluate(test_images,test_labels,verbose=1)
# test_loss,test_acc = loaded_model.evaluate(test_images,test_labels,verbose=1)
# print(test_loss)
# print("test_acc",test_acc)

# 手写数字图像识别
# 加载图片
written_image = cv2.imread("9b.png")
written_image = cv2.cvtColor(written_image, cv2.COLOR_BGR2GRAY)
written_image = cv2.resize(written_image, (28, 28))
plt.imshow(written_image, cmap=plt.cm.binary)
plt.show()
written_image = written_image.astype('float32') / 255
written_image = written_image.reshape((1, 28 * 28))
res = network.predict(written_image) 
print("预测的概率分布:", res[0])
# 获取预测结果（数字标签）
predicted_number = np.argmax(res[0])  # 返回最大值的索引
print("图片数字推理出来是：",predicted_number)

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
index = 20
digit = test_images[index]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
test_images = test_images.reshape((10000,28*28))
res = network.predict(test_images)
for i in range(res[index].shape[0]):
    if(res[index][i]==1):
        # print(res[index])
        print("图片数字推理出来是：",i)
        break;

