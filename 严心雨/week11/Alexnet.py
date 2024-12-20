#-----------------------------------------Alexnet_datast_process---------------------------------------------#
import os

# os.listdir(path):返回指定路径下的文件和文件夹列表
photos = os.listdir('E:/YAN/HelloWorld/cv/【11】CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/data/image/train/')

with open('E:/YAN/HelloWorld/cv/【11】CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/data/dataset','w') as f:
    for photo in photos:
        name = photo.split('.')[0]
        if name == 'cat':
            f.write(photo+';0\n')
        elif name == 'dog':
            f.write(photo+';1\n')
f.close()
#-----------------------------------------Alexnet_utils------------------------------------------------------#
import matplotlib.image as mping
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import numpy as np


# 将图片修剪成中心的正方形
def load_image(path):
    image = mping.imread(path)
    short_edge = min(image.shape[:2])
    yy = int((image.shape[0]-short_edge)/2) #记得要int。切割的图片大小必须是整型
    xx = int((image.shape[1]-short_edge)/2)
    crop_image = image[yy:yy+short_edge,xx:xx+short_edge]
    # plt.imshow(crop_image)
    # plt.show()
    return crop_image

def resize_image(image,size):
    #创建一个名称范围
    # with tf.compat.v1.name_scope('resize_image'):
    with tf.name_scope('resize_image'):
        #在这个名称范围下创建操作
        images = []
        for i in image:
            i = cv2.resize(i,size) # cv2.resize() 函数中dsize是二维的，不能是三维
            images.append(i)
        images = np.array(images)
        # print(images.shape)
        return images

def print_answer(argmax):
    with open('E:/YAN/HelloWorld/cv/[11]CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/data/model/index_word.txt','r',encoding='utf-8') as f:
        for l in f.readlines():
            synset = l.split(';')[1]
        print(answer)
    return synset[argmax]


if __name__=='__main__':
    # load_image('E:/YAN/HelloWorld/cv/【2】数学&数字图像/lenna.png')
    img = cv2.imread('lenna.png')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img / 255
    img_nor = np.expand_dims(img, axis=0)
    print(img_nor.shape)
    resize_image(img_nor, (224,224))

#-------------------------------------------------------Alexnet------------------------------------------------------------------------------------
import tensorflow
from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization


def AlexNet(input_shape=(224,224,3),output_shape=2):
    # AlexNet
    model = Sequential()
    # 构建卷积
    # 第一层卷积
    # Conv2D()：实现2维卷积操作
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层，因为将每个卷积层的filter减半了
    model.add(
        Conv2D(
            filters=48,
            kenel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    # 减均值除方差做归一化，提升收敛速度以及准确度
    # 在原版的AlexNet 是没有这一步的，因为上一步将每个卷积层的fliter减半了，效果减弱了，为了效果能更好一点，速度能快一点，
    # 所以加了这个
    model.add(BatchNormalization)
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    # 所建模型后输出为48特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid',
             )
    )
    #第二层卷积
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kenel_size=(5, 5),
            strides=(1, 1),
            padding='valid',
            activation='relu'
        )
    )
    #归一化
    model.add(BatchNormalization)
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        MaxPooling2D(
            pool_size=(3, 3),
            strides=(2, 2),
            padding='valid',
        )
    )
    #第三层卷积
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    # tf的padding有两个值，一个是SAME,一个是VALID。如果padding设置为SAME，则说明输入图片大小和输出图片大小是一致的，
    # 如果是VALID则图片经过滤波器后可能会变小
    model.add(
        Conv2D(
            filters=192,
            kenel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为384层，输出的shape为(13,13,384)；
    # 所建模型后输出为192特征层
    model.add(
        Conv2D(
            filters=192,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为1x1，大小为3的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128,
            kernel_size=(3,3),
            strides=(1,1),
            padding='same',
            activation='relu'
        )
    )
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(6,6,256)；
    # 所建模型后输出为128特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
        )
    )

    # 进入第一层全连接
    # 两个全连接层，最后输出为1000类,这里改为2类（猫和狗）
    # 输入缩减为1024
    # Flatten()：拍扁，把输入变成一维的
    # Dropout(0.25)：保留概率是0.75
    model.add(Flatten())
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    # 第二层全连接
    model.add(Dense(1024,activation='relu'))
    model.add(Dropout(0.25))
    # 输出
    model.add(Dense(output_shape, activation='softmax'))

    return model
#--------------------------------------------------------------------Alexnet_train----------------------------------------------------#
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
import keras.utils as np_utils
from keras.optimizers import Adam
from model.AlexNet_yan import AlexNet
import numpy as np
import AlexNet_utils
import cv2
from keras import backend as k

k.image_data_format() == 'channel_first' #通道数排最前面

def generate_arrays_from_file(lines,batch_size):
    #获取总长度-图片总量
    n = len(lines)
    i = 0
    while 1:
        x_train = []
        y_train = []
        for b in range(batch_size):
            if i == 0:
                #打乱
                np.random.shuffle(lines)
            #获取图片名字
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread('E:/YAN/HelloWorld/cv/[11]CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/data/image/train'+ '/' +name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            x_train.append(img)
            y_train.append(lines[i].split(';')[1])
            #读完一个周期后重新开始
            i = (i+1) % n

        x_train = AlexNet_utils.resize_image(x_train,(224,224))
        x_train = x_train.reshape(-1,224,224,3)#拍扁
        y_train = np_utils.to_categorical(np.array(y_train),num_classes=2)

        yield (x_train,y_train)

if __name__=='__main__':
    # 模型保存的位置
    log_dir = 'E:/YAN/HelloWorld/cv/[11]CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/logs/'
    # 打开数据集的txt
    with open(r'E:/YAN/HelloWorld/cv/[11]CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/data/dataset','r') as f:
        # 以每一行去读取标签文件
        lines = f.readlines()
    # 打乱行，这个用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    # random.seed()是设置的随机种子，保证每次运行代码随机化的顺序一致，减少不必要的随机性。
    # random.seed(something) 只能是一次有效。如果使用相同的seed(something)值，则每次生成的随机数都相同，
    # 如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同

    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines) * 0.1)
    num_train = len(lines)-num_val

    # 建立AlexNet模型，调用了建好了的AlexNet
    model = AlexNet()

    # 保存的方式，3代保存一次。
    # 3个epoch保存一次权重。目的，假如训练工程中因不明原因中断了，还可以继续训练
    checkpoint_period1 = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',# 地址的设定。存在哪，以及存取的文件名字
                                         monitor='acc',
                                         save_best_only=True,
                                         save_weights_only=False,
                                         period=3
                                         )
    # 学习率下降的方式，weight三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(monitor='acc',
                                  factor=0.5, # 每次乘0.5
                                  patience=3,# 等待3次，如果3次weight没下降，那就下降学习率继续训练
                                  verbose=1
                                  )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(monitor = 'acc',
                                   min_delta=0,  # 最小值是0
                                   patience=10,  # 做10次，如果loss 都不下降，训练就停止了
                                   verbose=1
                                   )
    # 损失函数-交叉熵
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = Adam(lr=1e-3),
                  metrics = ['accuracy'] # 准确度计算方式。评价函数用于苹果当前训练模型的性能
                 )

    # 一次的训练集大小
    batch_size = 128
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train],batch_size), #训练数据
                        steps_per_epoch=max(1,num_train // batch_size),
                        validation_data=generate_arrays_from_file(lines[num_train:],batch_size),
                        validation_steps=max(1,num_val // batch_size),
                        epochs=50,
                        initial_epoch=0,
                        callbacks=[checkpoint_period1,reduce_lr]#增加的额外功能
                        )
    model.save_weights(log_dir+'last1.h5')
  #------------------------------------------------------------Alexnet_predict------------------------------------------------------
  from model.AlexNet_yan import AlexNet
import AlexNet_utils
import cv2
from keras import backend as k
import numpy as np

k.image_data_format()=='channels_first'

if __name__=='__main__':
    model = AlexNet()
    model.load_weight('E:/YAN/HelloWorld/cv/【11】CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master/logs/last1.h5')
    img = cv2.imread('E:/YAN/HelloWorld/cv/【11】CNN/alexnet/AlexNet-Keras-master/AlexNet-Keras-master//test2.jpg')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img = img / 255
    img_nor = np.expand_dims(img,axis=0)
    img_nor = AlexNet_utils.resize_image(img_nor,(224,224))
    print('the answer is:',AlexNet_utils.print_answer(np.argmax(model.predict(img_nor))))
    # model是AlexNet的返回结果，而AlexNet返回的是keras里的Sequentrial类对象，里面包括predict方法
    # predict是Sequentrial等模型结构类定义的通用方法，会自动读取你定义的网络结构如卷积层这些，根据传的输入运行前向传播，
    # 跟是否训练无关，区别只是模型参数不一样而已
    cv2.imshow('ooo',img)
    cv2.waitKey(0)
