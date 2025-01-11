import numpy as np
import cv2
from keras.utils import np_utils

from 成元林.第十一周.AlexNet import utils
from 成元林.第十一周.AlexNet.model.AlexModelOfKeras import AlexNet_cat_dog_Model
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras import backend as K

K.set_image_data_format('channels_last') #返回默认图像的维度顺序，将图像的通道数放在返回参数中的最后

def getImageDataByName(image_name_list, batchSize):
    """
    根据文件名称获取图片
    @param image_name_list: 图片名称列表
    @param batchSize: 批次大小
    @return:
    """
    # 图片总量
    # 获取总长度
    n = len(image_name_list)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batchSize):  # batch中的每一个数据处理后加入XY两个数组中
            if i == 0:
                np.random.shuffle(image_name_list)  # 当i = 0时，为每个batch的第一个数据，此时打乱lines中的数据
            name = image_name_list[i].split(';')[0]  # 提取出txt文件中的;之前的名字，即为猫或狗的图片名称
            # 从文件中读取图像
            img = cv2.imread(r"./train" + '\\' + name)  # opencv读入为BGR，值为0-255
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像转为RGB图像
            img = img / 255  # 全部变为0-1之间的值
            X_train.append(img)  # X_train中加入img的所有像素值，数量和Y_train相等
            Y_train.append(image_name_list[i].split(';')[1])  # Y_train中为原数据中的；后的值（0或1）
            i = (i + 1) % n  # 读完一个batch周期后重新开始
        # 处理图像
        X_train = utils.resize_image(X_train, (224, 224))  # 将X_trian中每个图像重定义为224*224大小，用于使用AlexNet模型进行训练
        X_train = X_train.reshape(-1, 224, 224, 3)  # -1为通配符，python自动识别224*224*3的数据个数
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=2)  # karas中函数，用于设值2个类别（0，1）和（1，0）作为猫狗分类
        #生成器函数使用yield语句而不是return，这使得它们能够逐个产生值，而不是一次性生成整个序列。
        # 这种特性使得生成器在处理大量数据时非常高效，因为它们不需要将所有结果存储在内存中
        yield (X_train, Y_train)



def train(logpath):
    # 从txt读取文件内容
    with open(r"./data/dataset.txt", "r") as f:
        readlines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(readlines)
    np.random.seed(None)

    # 数据一部分作为训练数据，一部分作为测试数据
    num_val = int(len(readlines) * 0.1)
    num_train = len(readlines) - num_val

    # 创建模型
    model = AlexNet_cat_dog_Model()

    # 3代保存一次，保存最好的，不只是保存权重
    checkpoint_period1 = ModelCheckpoint(filepath=logpath + "loss{loss:.3f}.h5", monitor="acc", save_weights_only=False,
                                         save_best_only=True, period=3)

    reduce_lr = ReduceLROnPlateau(
        monitor="acc",
        factor=0.5,  # 缩放学习率的值，每次*1/2
        patience=10,  # 当3个epoch过后性能不再提升，则下降学习率
        verbose=1)

    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
        monitor="val_loss",#检测值为验证集的损失函数
        min_delta=0,#对于小于0的变化不关心，即为所有变化都会影响是否提前停止
        patience=10,#10个epoch都没有效率提升则结束
        verbose=1
    )

    # 交叉熵，编译创建好的模型，网络模型搭建完后，需要对网络的学习过程进行配置
    # 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准，均为karas自带
    model.compile(loss='categorical_crossentropy',
                  optimizer='sgd',
                  metrics=['accuracy'])

    # 一次的训练集大小
    batch_size = 128
    # 输出为训练集个数，验证集个数，一次的训练集大小
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练，利用Python的生成器，逐个生成数据的batch并进行训练
    # karas中sequence生成器（generator）
    model.fit_generator(getImageDataByName(readlines[:num_train], batch_size),  # lines为百分之九十数据集（训练集）
                        steps_per_epoch=max(1, num_train // batch_size),  # 每个epoch中需要执行多少次generator，最少1次，//为整除算法
                        validation_data=getImageDataByName(readlines[num_train:], batch_size),
                        # 用于验证，不参与计算，百分之十（验证集）
                        validation_steps=max(1, num_val // batch_size),  # 每个epoch中执行多少次generator
                        epochs=50,  # 迭代次数（世代）
                        initial_epoch=0,
                        callbacks=[checkpoint_period1, reduce_lr])  # 训练时用之前设定好的的回调函数
    model.save_weights(logpath + 'last1.h5')
if __name__ == '__main__':
    train("./logs/")