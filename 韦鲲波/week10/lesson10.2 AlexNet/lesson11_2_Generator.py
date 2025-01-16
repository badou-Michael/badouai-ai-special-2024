import numpy as np
import os
import cv2
from keras.utils import np_utils


def generator(pipeline, batch_size):

    while True:
        np.random.shuffle(pipeline)
        train = []
        label = []

        for i in range(batch_size):
            # 读取出文件里的文件名和标签
            train_img , label_img = pipeline[i].split(';')[0], pipeline[i].split(';')[1]

            # 根据文件名提取指定文件
            train_img = cv2.imread(os.path.join(r'alexnet\train', train_img))
            train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)

            # 把图片resize成227，再归一化
            train_img = cv2.resize(train_img, (227, 227)) / 255

            # 训练数据和标签数据存入列表
            train.append(train_img)
            label.append(label_img)

        # 列表改为np数组，训练数据reshape到100个batch上，标签进行one-hot处理
        train = np.reshape(train, (-1, 227, 227, 3))
        label = np_utils.to_categorical(np.array(label), 2)

        yield (train, label)


if __name__ == '__main__':
    with open(r'alexnet\AlexNet-Keras-master\data\dataset.txt', 'r') as f:
        pipeline = f.readlines()
    # a = []
    # for i in range(10):
    #     label_img = pipeline[0].split(';')[1]
    #     a.append(label_img)
    # label = np_utils.to_categorical(np.array(a), 2)
    # print(label)
    a = generator(pipeline, batch_size=100)
