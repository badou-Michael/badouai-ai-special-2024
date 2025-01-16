import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models
from tensorflow.keras import layers

[1]#导入数据

(train_images,train_labels),(test_images,test_labels)=mnist.load_data() #导入训练图片和标签，测试图片和标签


[2]#构建模型

#用sequential构建连续模型
network = models.Sequential()

#添加一个dense隐藏层，512个节点，每个节点接收长度为28*28的输入向量
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))

#添加一个dense输出层，共有10个节点，用softmax激活函数(把输出转成概率)
network.add(layers.Dense(10,activation='softmax'))

#用compile汇编方法告诉模型如何运作，优化框架用梯度下降优化算法，损失函数用交叉熵，标准度量用正确率
network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
                metrics=['accuracy'])

[3]#数据归一处理

#把数据从(60000,28,28)变成(60000,28*28)，28,28二维变成28*28的一维数组
train_images = train_images.reshape(60000, 28*28)
#归一化成浮点数
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape(10000, 28*28)
test_images = test_images.astype('float32') / 255

[4]#对标签one-hot编码

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

[5]#拟合训练

network.fit(train_images,train_labels, epochs=5,batch_size=100)

[6]#评测模型好坏

#评价模型输出损失loss和正确率acc
test_loss,test_acc =network.evaluate(test_images,test_labels, verbose=1)

[7]#传入一张照片测试效果

#把测试照片进行推理
res = network.predict(test_images)

#用argmax找出第二张测试照片的预测概率最高的数字，应该是2
predicted_class = np.argmax(res[1])
print('the number in picture is : ',predicted_class)
print(res)
