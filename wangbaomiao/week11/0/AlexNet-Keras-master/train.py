# -*- coding: utf-8 -*-
# time: 2024/11/19 16:20
# file: train.py
# author: flame
import cv2
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import Adam
from keras.utils import np_utils

import utils
from model.AlexNet import AlexNet

'''在 Keras 中，'tf' 是一个特殊的字符串，表示使用 TensorFlow 的默认格式，即 'channels_last'。'''
K.set_image_data_format('channels_last')

'''
此脚本从文件中读取训练数据，生成训练和验证数据的批次，并使用 AlexNet 模型进行训练。
主要步骤包括：
1. 从文件中读取训练数据行。
2. 将数据随机打乱并划分为训练集和验证集。
3. 定义模型并配置回调函数。
4. 使用生成器生成的数据训练模型。
5. 保存训练后的模型权重。
'''

''' 定义从文件中生成训练数据批次的函数。 '''
def generate_arrays_from_file(lines, batch_size):
    ''' 获取数据行数。 '''
    n = len(lines)
    ''' 初始化行索引。 '''
    i = 0
    ''' 无限循环以生成训练批次。 '''
    while 1:
        ''' 初始化训练数据和标签的列表。 '''
        X_train = []
        Y_train = []
        ''' 遍历批次大小以加载数据。 '''
        for b in range(batch_size):
            ''' 当索引归零时，重新打乱数据顺序，确保数据的随机性。 '''
            if i == 0:
                np.random.shuffle(lines)
            ''' 解析图像文件名。 '''
            name = lines[i].split(';')[0]
            ''' 读取图像数据。 '''
            img = cv2.imread(r".\data\image\train" + '/' + name)
            ''' 将图像从 BGR 格式转换为 RGB 格式。 '''
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ''' 将图像像素值归一化到 [0, 1] 范围。 '''
            img = img / 255
            ''' 将处理后的图像添加到训练数据列表中。 '''
            X_train.append(img)
            ''' 解析并添加标签到训练标签列表中。 '''
            Y_train.append(lines[i].split(';')[1])
            ''' 更新行索引，确保循环遍历所有数据。 '''
            i = (i + 1) % n
        ''' 调整图像尺寸为 224x224。 '''
        X_train = utils.resize_image(X_train, (224, 224))
        ''' 重塑训练数据的形状。 '''
        X_train = X_train.reshape(-1, 224, 224, 3)
        ''' 将标签转换为 one-hot 编码。 '''
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)
        ''' 生成训练数据和标签的元组。 '''
        yield (X_train, Y_train)

if __name__ == '__main__':
    ''' 定义日志目录路径。 '''
    log_dir = './logs/'
    ''' 从文件中读取训练数据行。 '''
    with open(r".\data\dataset.txt", "r") as f:
        lines = f.readlines()

    ''' 设置随机种子以确保数据打乱的可重复性。 '''
    np.random.seed(10101)
    ''' 打乱数据顺序，确保数据的随机性。 '''
    np.random.shuffle(lines)
    ''' 重置随机种子。 '''
    np.random.seed(None)

    ''' 计算验证集的大小，占总数据的 10%。 '''
    num_val = int(len(lines) * 0.1)
    ''' 计算训练集的大小。 '''
    num_train = len(lines) - num_val

    ''' 定义并编译 AlexNet 模型。 '''
    model = AlexNet()

    ''' 定义模型检查点回调，每 3 个周期保存一次模型。 '''
    checkpoint_period1 = ModelCheckpoint(
        log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
        monitor='acc',
        save_weights_only=False,
        save_best_only=True,
        period=3
    )

    ''' 定义学习率衰减回调，当准确率不再提高时降低学习率。 '''
    reduce_lr = ReduceLROnPlateau(
        monitor='acc',
        factor=0.1,
        patience=3,
        verbose=1
    )

    ''' 定义提前停止回调，当准确率不再提高时提前停止训练。 '''
    early_stopping = EarlyStopping(
        monitor='acc',
        min_delta=0,
        patience=10,
        verbose=1
    )

    ''' 编译模型，指定损失函数、优化器和评估指标。 '''
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(lr=1e-3),
        metrics=['accuracy']
    )

    ''' 定义批量大小。 '''
    batch_size = 128

    ''' 打印训练和验证样本数以及批量大小。 '''
    print('训练样本数: {}, 验证样本数: {}, 批量大小: {}.'.format(num_train, num_val, batch_size))

    ''' 使用生成器训练模型。 '''
    model.fit_generator(
        generate_arrays_from_file(lines[:num_train], batch_size),
        steps_per_epoch=max(1, num_train // batch_size),
        validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
        validation_steps=max(1, num_val // batch_size),
        epochs=50,
        initial_epoch=0,
        callbacks=[checkpoint_period1, reduce_lr]
    )

    ''' 保存训练后的模型权重。 '''
    model.save_weights(log_dir + 'last_weights.h5')
