#读取数据
from tensorflow.keras.datasets import mnist
(train_image,train_label),(test_image,test_label)=mnist.load_data()
#查看训练集和验证集数据的shape
print('train_image.shape',train_image.shape) #结果是60000，28*28
print('test_image.shape = ', test_image.shape) #结果是10000,28*28


#搭建神经网络结构
from tensorflow.keras import models
from tensorflow.keras import layers
#Sequential 是一个用于创建顺序模型的类,是一种线性堆叠的神经网络层，其中每一层的输出都会成为下一层的输入
network=models.Sequential()
#构建隐藏层
network.add(layers.Dense(256,activation='relu',input_shape=(28*28,)))
#构建输出层
network.add(layers.Dense(10,activation='softmax'))
#设置参数
network.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

#调整图片的格式由二维改成一维
train_image=train_image.reshape((60000,28*28))
train_image=train_image.astype('float32')/255
test_image=test_image.reshape((10000,28*28))
test_image=test_image.astype('float32')/255
#调整输出结果，将具体数字改为0/1数组
from tensorflow.keras.utils import to_categorical
train_label=to_categorical(train_label)
test_label=to_categorical(test_label)

#训练拟合数据
network.fit(train_image,train_label,epochs=5,batch_size=256)
#验证结果
test_loss,test_acc=network.evaluate(test_image,test_label,verbose=2)
print('test_loss',test_loss)
print('test_acc',test_acc)

#商用测试，以验证集中的第一张图片作为商用测试
#同时画出图片，打印出预测值，人眼对比是否正确
(train_image,train_label),(test_image,test_label)=mnist.load_data()
digit=test_image[1]
import matplotlib.pyplot as plt
plt.imshow(digit)
plt.show()
test_image=test_image.reshape((10000,28*28))
res=network.predict(test_image)

for i in range(res[1].shape[0]):
    if (res[1][i]==1):
        print('predict result is:',i)
        break
