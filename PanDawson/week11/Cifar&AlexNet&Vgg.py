#该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 
import numpy as np
import time
import math

num_classes=10

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

#定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


#定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue):
    result=CIFAR10Record()

    label_bytes=1                                            #如果是Cifar-100数据集，则此处为2
    result.height=32
    result.width=32
    result.depth=3                                           #因为是RGB三通道，所以深度是3

    image_bytes=result.height * result.width * result.depth  #图片样本总元素数量
    record_bytes=label_bytes + image_bytes                   #因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)  #使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    result.key,value=reader.read(file_queue)                 #使用该类的read()函数从文件队列里面读取文件

    record_bytes=tf.decode_raw(value,tf.uint8)               #读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    
    #因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)

    #剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    #这一步是将一维数据转换成3维数据
    depth_major=tf.reshape(tf.strided_slice(record_bytes,[label_bytes],[label_bytes + image_bytes]),
                           [result.depth,result.height,result.width])  

    #我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    #这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image=tf.transpose(depth_major,[1,2,0])

    return result                                 #返回值是已经把目标文件里面的信息都读取出来

def inputs(data_dir,batch_size,distorted):               #这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
    filenames=[os.path.join(data_dir,"data_batch_%d.bin"%i)for i in range(1,6)]   #拼接地址

    file_queue=tf.train.string_input_producer(filenames)     #根据已经有的文件地址创建一个文件队列
    read_input=read_cifar10(file_queue)                      #根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image=tf.cast(read_input.uint8image,tf.float32)   #将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch=num_examples_pre_epoch_for_train


    if distorted != None:                         #如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
        cropped_image=tf.random_crop(reshaped_image,[24,24,3])          #首先将预处理好的图片进行剪切，使用tf.random_crop()函数

        flipped_image=tf.image.random_flip_left_right(cropped_image)    #将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数

        adjusted_brightness=tf.image.random_brightness(flipped_image,max_delta=0.8)   #将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数

        adjusted_contrast=tf.image.random_contrast(adjusted_brightness,lower=0.2,upper=1.8)    #将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数

        float_image=tf.image.per_image_standardization(adjusted_contrast)          #进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差

        float_image.set_shape([24,24,3])                      #设置图片数据及标签的形状
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)

        images_train,labels_train=tf.train.shuffle_batch([float_image,read_input.label],batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )
                             #使用tf.train.shuffle_batch()函数随机产生一个batch的image和label

        return images_train,tf.reshape(labels_train,[batch_size])

    else:                               #不对图像数据进行数据增强处理
        resized_image=tf.image.resize_image_with_crop_or_pad(reshaped_image,24,24)   #在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切

        float_image=tf.image.per_image_standardization(resized_image)          #剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples=int(num_examples_per_epoch * 0.4)

        images_test,labels_test=tf.train.batch([float_image,read_input.label],
                                              batch_size=batch_size,num_threads=16,
                                              capacity=min_queue_examples + 3 * batch_size)
                                 #这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test,tf.reshape(labels_test,[batch_size])





#该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程


max_steps=4000
batch_size=100
num_examples_for_eval=10000
data_dir="Cifar_data/cifar-10-batches-bin"

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    #truncated_normal 函数生成的是截断正态分布的随机数，相比于普通的正态分布，它会截断那些距离均值超过 2 倍标准差的样本，重新在区间内采样，
    #这样做的好处是可以避免生成一些过大或过小的极端值，使得初始化的参数相对更合理、更稳定，尤其是在神经网络训练初期，有助于训练过程更平稳地进行。
    #shape：这是一个必需的参数，用于指定要生成的随机数张量的形状。例如，如果要初始化一个神经网络中某层权重矩阵，其形状可能是 [输入层节点数量, 隐藏层节点数量]
    #那就通过 shape 参数传入对应的维度值（以列表或者元组形式，如 (input_size, hidden_size)），来确定最终生成的随机数张量的维度
    #stddev 是一个可选参数，用于指定生成的正态分布随机数的标准差，控制生成的随机数的离散程度
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

#使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
#其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
images_train,labels_train=inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
images_test,labels_test=inputs(data_dir=data_dir,batch_size=batch_size,distorted=None)

#创建x和y_两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值。
#要注意的是，由于以后定义全连接网络的时候用到了batch_size，所以x中，第一个参数不应该是None，而应该是batch_size
x=tf.placeholder(tf.float32,[batch_size,24,24,3])
y_=tf.placeholder(tf.int32,[batch_size])

#创建第一个卷积层 shape= [filter_height, filter_width, in_channels, out_channels]
kernel1=variable_with_weight_loss(shape=[5,5,3,64],stddev=5e-2,w1=0.0)
# x 输入的数据,[batch_size, height, width, channels]
#[1,1,1,1]  strides 参数  batch 维度、height 维度、width 维度以及 channels 维度上的步长
#padding 自动计算在输入数据的边缘需要填充多少个像素
conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding="SAME")
bias1=tf.Variable(tf.constant(0.0,shape=[64]))
relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))

pool1=tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#创建第二个卷积层
kernel2=variable_with_weight_loss(shape=[5,5,64,64],stddev=5e-2,w1=0.0)
conv2=tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding="SAME")
bias2=tf.Variable(tf.constant(0.1,shape=[64]))
relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2=tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding="SAME")

#因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，并使用get_shape()函数获取扁平化之后的长度
reshape=tf.reshape(pool2,[batch_size,-1])    #这里面的-1代表将pool2的三维结构拉直为一维结构
dim=reshape.get_shape()[1]             #get_shape()[1].value表示获取reshape之后的第二个维度的值

#建立第一个全连接层
weight1=variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1=tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#建立第二个全连接层
weight2=variable_with_weight_loss(shape=[384,192],stddev=0.04,w1=0.004)
fc_bias2=tf.Variable(tf.constant(0.1,shape=[192]))
local4=tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#建立第三个全连接层 tf.matmul
weight3=variable_with_weight_loss(shape=[192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3=tf.Variable(tf.constant(0.1,shape=[10]))
result=tf.add(tf.matmul(local4,weight3),fc_bias3)

#计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y_,tf.int64))

weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss=tf.reduce_mean(cross_entropy)+weights_with_l2_loss

train_op=tf.train.AdamOptimizer(1e-3).minimize(loss)

#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op=tf.nn.in_top_k(result,y_,1)

init_op=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()      

#每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range (max_steps):
        start_time=time.time()
        image_batch,label_batch=sess.run([images_train,labels_train])
        _,loss_value=sess.run([train_op,loss],feed_dict={x:image_batch,y_:label_batch})
        duration=time.time() - start_time

        if step % 100 == 0:
            examples_per_sec=batch_size / duration
            sec_per_batch=float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

#计算最终的正确率
    num_batch=int(math.ceil(num_examples_for_eval/batch_size))  #math.ceil()函数用于求整
    true_count=0
    total_sample_count=num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch,label_batch=sess.run([images_test,labels_test])
        predictions=sess.run([top_k_op],feed_dict={x:image_batch,y_:label_batch})
        true_count += np.sum(predictions)

    #打印正确率信息
    print("accuracy = %.3f%%"%((true_count/total_sample_count) * 100))


# step 0,loss=4.68(36.0 examples/sec;2.776 sec/batch)
# step 100,loss=1.96(4427.9 examples/sec;0.023 sec/batch)
# step 200,loss=1.76(4150.7 examples/sec;0.024 sec/batch)
# step 300,loss=1.75(4174.1 examples/sec;0.024 sec/batch)
# step 400,loss=1.75(4341.3 examples/sec;0.023 sec/batch)
# step 500,loss=1.35(4213.6 examples/sec;0.024 sec/batch)
# step 600,loss=1.64(4118.7 examples/sec;0.024 sec/batch)
# step 700,loss=1.37(4138.7 examples/sec;0.024 sec/batch)
# step 800,loss=1.68(4343.0 examples/sec;0.023 sec/batch)
# step 900,loss=1.22(4235.4 examples/sec;0.024 sec/batch)
# step 1000,loss=1.26(4385.9 examples/sec;0.023 sec/batch)
# step 1100,loss=1.22(4250.3 examples/sec;0.024 sec/batch)
# step 1200,loss=1.39(4376.5 examples/sec;0.023 sec/batch)
# step 1300,loss=1.41(4331.0 examples/sec;0.023 sec/batch)
# step 1400,loss=1.32(4246.0 examples/sec;0.024 sec/batch)
# step 1500,loss=1.28(4299.2 examples/sec;0.023 sec/batch)
# step 1600,loss=1.32(3747.0 examples/sec;0.027 sec/batch)
# step 1700,loss=1.35(4268.6 examples/sec;0.023 sec/batch)
# step 1800,loss=1.19(4272.0 examples/sec;0.023 sec/batch)
# step 1900,loss=0.96(4336.3 examples/sec;0.023 sec/batch)
# step 2000,loss=1.10(4585.3 examples/sec;0.022 sec/batch)
# step 2100,loss=1.14(4387.0 examples/sec;0.023 sec/batch)
# step 2200,loss=1.14(4274.8 examples/sec;0.023 sec/batch)
# step 2300,loss=1.14(4039.0 examples/sec;0.025 sec/batch)
# step 2400,loss=1.28(4148.0 examples/sec;0.024 sec/batch)
# step 2500,loss=1.30(3565.3 examples/sec;0.028 sec/batch)
# step 2600,loss=1.05(4361.9 examples/sec;0.023 sec/batch)
# step 2700,loss=1.14(4311.1 examples/sec;0.023 sec/batch)
# step 2800,loss=1.34(4299.3 examples/sec;0.023 sec/batch)
# step 2900,loss=1.20(4206.5 examples/sec;0.024 sec/batch)
# step 3000,loss=1.11(4246.7 examples/sec;0.024 sec/batch)
# step 3100,loss=1.18(4402.9 examples/sec;0.023 sec/batch)
# step 3200,loss=0.99(4239.8 examples/sec;0.024 sec/batch)
# step 3300,loss=1.14(4305.1 examples/sec;0.023 sec/batch)
# step 3400,loss=0.96(4231.0 examples/sec;0.024 sec/batch)
# step 3500,loss=1.11(4176.3 examples/sec;0.024 sec/batch)
# step 3600,loss=1.11(4206.6 examples/sec;0.024 sec/batch)
# step 3700,loss=0.84(4167.8 examples/sec;0.024 sec/batch)
# step 3800,loss=1.17(4229.2 examples/sec;0.024 sec/batch)
# step 3900,loss=1.04(4387.8 examples/sec;0.023 sec/batch)
# accuracy = 71.640%


#AlexNet
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 
from tensorflow.python.ops import array_ops
import matplotlib.image as mpimg
import numpy as np
import cv2

from keras.models import Sequential
from keras.layers import Dense,Activation,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization
from keras.datasets import mnist
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.utils import np_utils
from keras import backend as K

def load_image(path):
    # 读取图片，rgb
    img = mpimg.imread(path)
    # 将图片修剪成中心的正方形
    #元组中的前两个元素，即图像的高度和宽度信息
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    return crop_img

def resize_image(image, size):
    with tf.name_scope('resize_image'):
        images = []
        for i in image:
            i = cv2.resize(i, size)
            images.append(i)
        images = np.array(images)
        return images

def print_answer(argmax):
    with open("./data/model/index_word.txt","r",encoding='utf-8') as f:
        synset = [l.split(";")[1][:-1] for l in f.readlines()]
        
    return synset[argmax]

def AlexNet(input_shape=(224,224,3),output_shape=2):
    # AlexNet
    model = Sequential()
    # 使用步长为4x4，大小为11的卷积核对图像进行卷积，输出的特征层为96层，输出的shape为(55,55,96)；
    # 所建模型后输出为48特征层
    model.add(
        Conv2D(
            filters=48, 
            kernel_size=(11,11),
            strides=(4,4),
            padding='valid',
            input_shape=input_shape,
            activation='relu'
        )
    )
    
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(27,27,96)
    # 所建模型后输出为48特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3), 
            strides=(2,2), 
            padding='valid'
        )
    )
    # 使用步长为1x1，大小为5的卷积核对图像进行卷积，输出的特征层为256层，输出的shape为(27,27,256)；
    # 所建模型后输出为128特征层
    model.add(
        Conv2D(
            filters=128, 
            kernel_size=(5,5), 
            strides=(1,1), 
            padding='same',
            activation='relu'
        )
    )
    
    model.add(BatchNormalization())
    # 使用步长为2的最大池化层进行池化，此时输出的shape为(13,13,256)；
    # 所建模型后输出为128特征层
    model.add(
        MaxPooling2D(
            pool_size=(3,3),
            strides=(2,2),
            padding='valid'
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
    # 两个全连接层，最后输出为1000类,这里改为2类（猫和狗）
    # 缩减为1024
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.25))
    
    model.add(Dense(output_shape, activation='softmax'))

    return model

K.image_data_format() == 'channels_first'

def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for b in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取图像
            img = cv2.imread(r"./data/image/train/train" + '/' + name)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img = img/255
            X_train.append(img)
            Y_train.append(lines[i].split(';')[1])
            # 读完一个周期后重新开始
            i = (i+1) % n
        # 处理图像
        X_train = resize_image(X_train,(224,224))
        # -1 通常对应着图像的批量大小
        X_train = X_train.reshape(-1,224,224,3)
        #独热编码
        Y_train = np_utils.to_categorical(np.array(Y_train),num_classes= 2)   
        yield (X_train, Y_train)

if __name__ == "__main__":
    # 模型保存的位置
    log_dir = "./logs/"

    # 打开数据集的txt
    with open(r"./data/dataset.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 建立AlexNet模型
    model = AlexNet()
    
    # 保存的方式，3代保存一次
    checkpoint_period1 = ModelCheckpoint(
                                    #log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    log_dir + 'checkpoint.h5',
                                    monitor='acc', 
                                    save_weights_only=False, 
                                    save_best_only=True, 
                                    save_freq="epoch"
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='acc', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss',  #损失值（Loss）变化 ；val_accuracy（验证集准确率，常用于分类任务）
                            min_delta=0,  #最小变化量。超过这个最小变化量时，才会被认为是有实质意义的变化
                            patience=10, #损失值都没有出现大于 min_delta 的改善情况，就会触发提前停止训练的机制
                            verbose=1 #用于控制在训练过程中相关信息的打印输出详细程度。 0 1 2。 1 时，表示会在触发提前停止训练操作或者相关判断过程中有比较详细的信息打印输出
                        )

    # 交叉熵
    model.compile(loss = 'categorical_crossentropy',
            #Adam 优化器在训练过程中还会根据梯度的一阶矩估计和二阶矩估计自适应地调整每个参数对应的学习率
            optimizer = Adam(lr=1e-3),
            ##用于指定在训练和验证过程中要计算和展示的评估指标
            metrics = ['accuracy'])

    # 一次的训练集大小
    batch_size = 128

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    #用于使用数据生成器（Generator）来训练模型——逐批次地从磁盘等外部存储读取数据并生成训练样本  fit_generator
    model.fit(generate_arrays_from_file(lines[:num_train], batch_size),
            #用于指定每个训练轮次（Epoch）中从数据生成器中获取数据的步数（也就是批次数量）
            steps_per_epoch=max(1, num_train//batch_size),
            #验证集
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            #验证集每一步多少数据
            validation_steps=max(1, num_val//batch_size),
            epochs=50,
            initial_epoch=0,
            #接收一个回调函数（或回调对象）列表。定期保存参数&学习率。
            callbacks=[checkpoint_period1, reduce_lr])
    model.save_weights(log_dir+'last1.h5')


# 173/175 [============================>.] - ETA: 1s - batch: 86.0000 - size: 128.0000 - loss: 0.4725 - accuracy: 0.7711
# 175/175 [==============================] - ETA: 0s - batch: 87.0000 - size: 128.0000 - loss: 0.4722 - accuracy: 0.7712
# 175/175 [==============================] - 151s 867ms/step - batch: 87.0000 - size: 128.0000 - loss: 0.4722 - accuracy: 0.7712 - val_loss: 0.4676 - val_accuracy: 0.7866 - lr: 0.0010

#VGG

import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution() 
import numpy as np
from tensorflow.python.ops import array_ops
import matplotlib.image as mpimg
import tf_slim as slim

def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # 将概率从大到小排列的结果的序号存入pred
    pred = np.argsort(prob)[::-1]
    # 取最大的1个、5个。
    top1 = synset[pred[0]]
    print(("Top1: ", top1, prob[pred[0]]))
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(("Top5: ", top5))
    return top1

slim = slim

def vgg_16(inputs,
           num_classes=1000,
           is_training=True,
           dropout_keep_prob=0.5,
           spatial_squeeze=True,
           scope='vgg_16'):

    with tf.variable_scope(scope, 'vgg_16', [inputs]):
        # 建立vgg_16的网络

        # conv1两次[3,3]卷积网络，输出的特征层为64，输出为(224,224,64)
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        # 2X2最大池化，输出net为(112,112,64)
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # conv2两次[3,3]卷积网络，输出的特征层为128，输出net为(112,112,128)
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        # 2X2最大池化，输出net为(56,56,128)
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(56,56,256)
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        # 2X2最大池化，输出net为(28,28,256)
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(28,28,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        # 2X2最大池化，输出net为(14,14,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # conv3三次[3,3]卷积网络，输出的特征层为256，输出net为(14,14,512)
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        # 2X2最大池化，输出net为(7,7,512)
        net = slim.max_pool2d(net, [2, 2], scope='pool5')

        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                            scope='dropout6')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,4096)
        net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                            scope='dropout7')
        # 利用卷积的方式模拟全连接层，效果等同，输出net为(1,1,1000)
        net = slim.conv2d(net, num_classes, [1, 1],
                        activation_fn=None,
                        normalizer_fn=None,
                        scope='fc8')
        
        # 由于用卷积的方式模拟全连接层，所以输出需要平铺
        if spatial_squeeze:
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        return net

img1 = load_image("./test_data/table.jpg")

# 对输入的图片进行resize，使其shape满足(-1,224,224,3)
inputs = tf.placeholder(tf.float32,[None,None,3])
resized_img = resize_image(inputs, (224, 224))

# 建立网络结构
prediction = vgg_16(resized_img)

# 载入模型
sess = tf.Session()
ckpt_filename = './model/vgg_16.ckpt'
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(sess, ckpt_filename)

# 最后结果进行softmax预测
pro = tf.nn.softmax(prediction)
pre = sess.run(pro,feed_dict={inputs:img1})

# 打印预测结果
print("result: ")
print_prob(pre[0], './synset.txt')

# ('Top1: ', 'n03201208 dining table, board', 0.999111)
# ('Top5: ', [('n03201208 dining table, board', 0.999111), ('n03376595 folding chair', 0.00052908185), ('n03179701 desk', 0.00024559672), ('n04099969 rocking chair, rocker', 5.9871036e-05), ('n03125729 cradle', 2.1230993e-05)])
# 'n03201208 dining table, board'
