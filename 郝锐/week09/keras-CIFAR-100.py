#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/18 16:14
# @Author: Gift
# @File  : keras-CIFAR-100.py 
# @IDE   : PyCharm
import keras
from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
"""
数据都是32*32的图像
"""
# print(y_train[0])
# 数据预处理-将像素值归一化
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
# 将标签转换为one-hot编码
y_train = to_categorical(y_train, 100)
y_test = to_categorical(y_test, 100)
# plt.imshow(x_train[0])
# plt.show()
# print(y_train[0])
#数据增强
datagen = ImageDataGenerator(
    rotation_range=15,  # 随机旋转图片的角度范围
    width_shift_range=0.1,  # 随机水平平移的范围
    height_shift_range=0.1,  # 随机垂直平移的范围
    horizontal_flip=True,  # 随机水平翻转图片
    fill_mode='nearest'  # 填充新创建像素的方法
    )
datagen.fit(x_train)
# 构建卷积神经网络模型
model = Sequential()
#添加一个卷积层，输入的形状为32，32，3,padding=same,表示在卷积过程中保持输入的尺寸不变，padding=valid表示在卷积过程中会缩小输入的尺寸
model.add(Conv2D(32, (3, 3), activation='relu', padding="same", input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', padding="same"))
model.add(BatchNormalization())
#添加一个最大池化层，池化窗口大小为2*2
model.add(MaxPooling2D(pool_size=(2, 2)))
#添加一个dropout层，防止过拟合
model.add(Dropout(0.25))
#添加第二个卷积层，
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#在每个卷积层之后添加了 BatchNormalization 层，它可以加速模型的训练收敛速度，
#减少梯度消失或爆炸问题，并且在一定程度上具有正则化的效果，有助于提高模型的性能和稳定性。
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))
#添加第三个卷积层
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
# 添加全连接层
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
# 添加输出层
model.add(Dense(100, activation='softmax'))
# 优化器
optimizer = Adam(learning_rate=0.001)
# 编译模型
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])
#调整学习率策略
# 学习率调整策略当验证损失在3个epoch中没有改善，则学习率再降低0.1，可以是模型更有效的收敛，避免陷入局部最优解
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.00001)
# 提前停止策略，当val_loss在5步中没有改善，则停止训练
early_stop = EarlyStopping(monitor='val_loss', patience=5)
# 训练模型
model.fit(
        datagen.flow(x_train, y_train, batch_size=64),
        steps_per_epoch=int(len(x_train) / 64),
        epochs=50,
        validation_data=(x_test, y_test),
        callbacks=[reduce_lr, early_stop]
)
# 在测试集上评估模型
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy:", scores[1])
print(model.summary())
