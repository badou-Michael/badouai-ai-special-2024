#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV 
@File    ：train.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/12/13 14:09
'''
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras.optimizers import Adam
from model.AlexNet import AlexNet
import numpy as np
import utils
import cv2
from keras import backend as K
# K.set_image_dim_ordering('tf')
K.image_data_format() == 'channels_first'


def generate_arrays_from_file(lines, batch_size):
    # 获取 lines 的总长度
    n = len(lines)
    i = 0
    while 1:
        # 存储图像数据的列表
        X_train = []
        # 存储标签数据的列表
        Y_train = []
        # 循环生成一个 batch_size 大小的数据
        for b in range(batch_size):
            if i == 0:
                # 打乱 lines 的顺序，确保每次读取数据的随机性
                np.random.shuffle(lines)
            # 从当前行中分割出图像的名称
            name = lines[i].split(';')[0]
            # 读取图像文件
            img = cv2.imread(r".\data\image\train" + '/' + name)
            # 将图像从 BGR 颜色空间转换为 RGB 颜色空间
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # 将图像像素值归一化到 0 到 1 之间
            img = img / 255
            # 将图像添加到 X_train 列表中
            X_train.append(img)
            # 从当前行中分割出标签并添加到 Y_train 列表中
            Y_train.append(lines[i].split(';')[1])
            # 更新 i 的值，循环读取 lines 中的数据
            i = (i + 1) % n
        # 调整图像的大小为 (224, 224)
        X_train = utils.resize_image(X_train, (224, 224))
        # 将图像数据重塑为 (batch_size, 224, 224, 3) 的形状
        X_train = X_train.reshape(-1, 224, 224, 3)
        # 将标签数据转换为 one-hot 编码，类别数为 2
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        # 使用 yield 关键字将数据作为生成器输出
        yield (X_train, Y_train)


if __name__ == "__main__":
    # 模型保存的目录
    log_dir = "./logs/"

    # 打开存储数据集信息的 txt 文件
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    # 设置随机数种子并打乱 lines 的顺序
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 划分训练集和验证集，90% 用于训练，10% 用于验证
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    # 建立 AlexNet 模型
    model = AlexNet()

    # 定义模型保存的回调函数，每 3 个 epoch 保存一次
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )
    # 定义学习率下降的回调函数，当 acc 三次不下降时，将学习率减半
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.5,
        patience=3,
        verbose=1
    )
    # 定义早停的回调函数，当 val_loss 连续 10 个 epoch 不下降时停止训练
    early_stopping = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1
    )

    # 编译模型，使用交叉熵作为损失函数，Adam 优化器，学习率为 1e-3，评估指标为准确率
    model.compile(loss='categorical_crossentropy',
                 optimizer=Adam(lr=1e-3),
                 metrics=['accuracy'])

    # 定义批次大小
    batch_size = 128

    # 打印训练和验证集的样本数量以及批次大小
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练模型
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
                      steps_per_epoch=max(1, num_train // batch_size),
                      validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
                      validation_steps=max(1, num_val // batch_size),
                      epochs=50,
                      initial_epoch=0,
                      callbacks=[checkpoint_period1, reduce_lr])
    # 保存最终的模型权重
    model.save_weights(log_dir + 'last1.h5')