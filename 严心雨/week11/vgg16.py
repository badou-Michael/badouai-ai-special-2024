#-----------------------------------------------------vgg16_utils-----------------------------------------------------------#
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
#-----------------------------------------------------vgg16-----------------------------------------------------------------#
import tensorflow as tf
# 创建slim对象
# 把tf.nn 里的接口都封装好，写的时候就简洁很多
# from tensorflow.contrib import slim
slim = tf.contrib.slim

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'
           ):

    """tf.variable_scope：用于定义变量（或层）的创建操作的上下文管理器。这个功能主要用于变量的共享和命名
       tf.variable_scope(scope, 'vgg_16', [inputs],reuse=reuse):
       scope:变量作用域的名字，如'vgg_16',用来组织相关的变量
       'vgg_16':如果设置了默认名称（在这里），则可以简化为仅传入作用域名，因为这是默认的子命名空间。
       [inputs]:可选的输入列表，这些输入可能会有与该作用作用域内的变量交互
       reuse:如果设为True,表示在同一个作用域内重用已经存在的变量，避免重复创建
    """
    with tf.variable_scope(scope,'vgg_16',[inputs]):
        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs,2,slim.conv2d,64,[3,3],scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net,[2,2],scope='pool1')

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net,2,slim.conv2d,128,[3,3],scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net,[2,2],scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net,3,slim.conv2d,256,[3,3],scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net,[2,2],scope='pool3')

        # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(28,28,512)
        net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net,[2,2],scope='pool4')

        # conv3三次[3,3]卷积网络，输出的特征层为512，输出net为(14,14,512)
        net = slim.repeat(net,3,slim.conv2d,512,[3,3],scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net,[2,2],scope='pool5')

        # 利用卷积的方式模拟全连接层，所以没有flatten了，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net,4096,[7,7],padding='VALID',scope='fc6')
        net = slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout6')

        # 利用卷积的方式模拟全连接层，所以没有flatten了，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net,4096,[1,1],scope='fc7')
        net = slim.dropout(net,dropout_keep_prob,is_training=is_training,scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        # activation_fn:用于激活函数的指定，默认为RELU函数
        # normalizer_fn:用于指定正则化函数
        net = slim.conv2d(net,num_classes, [1,1], activation_fn=None,normalizer_fn=None,scope='fc8')

        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net,[1,2],name='fc8/squeezed')# 删掉 第1维和第2维的1，那就只剩下1000了
        return net
#--------------------------------------------vgg16_train-------------------------------------------------#
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
    model = vgg_16()

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

#------------------------------------------------vgg16_demo-------------------------------------------------------#
from model import vgg16_yan
import tensorflow as tf
import vgg16_utils

#读取图片
img = vgg16_utils.load_image("E:/YAN/HelloWorld/homeworks/vgg16/test_data/table.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
#tf.reset_default_graph()
resized_img = vgg16_utils.resize_image(inputs, (224, 224))

# 建立网络结构
prediction = vgg16_yan.vgg_16(resized_img)

# 载入模型
sess = tf.Session()
ckpt_filename = 'E:/YAN/HelloWorld/cv/【11】CNN/vgg/vgg/VGG16-tensorflow-master/model/vgg_16.ckpt' # 训练模型时train模块
sess.run(tf.global_variables_initializer())
saver= tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img})

# 打印预测结果
print('result:')
vgg16_utils.print_prob(pre[0],'E:/YAN/HelloWorld/homeworks/vgg16/label/synset.txt') # 第0行的所有列数据









