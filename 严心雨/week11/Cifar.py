import os
import tensorflow as tf

#有10类图片
num_classes = 10

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

#定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass

def read_cifar10(file_queue):
    result = CIFAR10Record()
    result.label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    record_bytes = result.label_bytes + image_bytes

    """
    tf.FixedLengthRecordReader():是一个从文件中输出固定长度Recorder的类，可以将字符串（一般是一系列文件名）转化为Records
    (每个Recorder是一个Key,Value对)，Reader的每一步操作都会生成一个Record，这些Records都是从这些文件内容里提炼出来的。
    Reader可以通过Read()方法，从队列里列出一条记录Recorder。
    使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    """
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    key,value = reader.read(file_queue)

    # 读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组。uint8：无符号定点数
    record_bytes= tf.decode_raw(value,tf.uint8)

    #因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    #tf.strided_slice(record_bytes,[0],[label_bytes])：从record_bytes的第0位开始取，到label_bytes位结束的数据，即标签
    result.label = tf.cast(tf.strided_slice(record_bytes,[0],[result.label_bytes]),tf.int32)
    #剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * depth，我们要把这种格式转换成tensor-[depth,height,depth]
    #这一步是将一维数据转换成3维数据
    depth_image = tf.reshape(tf.strided_slice(record_bytes,[result.label_bytes],[result.label_bytes+image_bytes]),[result.depth,result.height,result.width])

    #我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    #这一步是转换数据排布方式，变为(h,w,c)
    #对应tf来说，这一步不必要，因为tf (h,w,c) /(c,h,w) 都适用。pytorch只能用(h,w,c)
    result.uint8image = tf.transpose(depth_image,[1,2,0])

    return result

#这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
def inputs(data_dir,batch_size,distorted):
    filenames = [os.path.join(data_dir,'data_batch_%d.bin' % i) for i in range(1,6)] #拼接地址

    #根据已经有的文件地址创建一个文件队列
    file_queue = tf.train.string_input_producer(filenames)
    #根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()
    read_input = read_cifar10(file_queue)
    #将已经转换好的图片数据再次转换为float32的形式
    reshaped_image = tf.cast(read_input.uint8image,tf.float32)

    num_examples_per_epoch = num_examples_pre_epoch_for_train

    # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理，提升图像质量
    if distorted != None:
        #首先将预处理好的图片进行剪切，使用tf.random_crop()函数，把它压缩成24*24*3的
        cropped_image = tf.random_crop(reshaped_image,[24,24,3])
        #左右翻转
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        #随机亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image,max_delta=0.8)
        #对比度调整
        adjusted_contrast = tf.image.adjust_contrast(adjusted_brightness,0.4)
        #标准化
        float_image = tf.image.per_image_standardization(adjusted_contrast)
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              %min_queue_examples)
        #使用tf.train.shuffle_batch()函数读取随机产生一个batch的image和label
        image_train,label_train = tf.train.shuffle_batch([float_image,read_input.label],
                                                         batch_size=batch_size,
                                                         num_threads=16,
                                                         capacity=min_queue_examples + 3 * batch_size,
                                                         min_after_dequeue=min_queue_examples,
                                                         )
        return image_train,tf.reshape(label_train,[batch_size])
    # 不对图像数据进行数据增强处理
    else:
        #裁剪
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image,[24,24])
        #标准化
        float_image = tf.image.per_image_standardization(resized_image)
        float_image.set_shape([24,24,3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        # 使用tf.train.shuffle_batch()函数读取随机产生一个batch的image和label
        image_test, label_test = tf.train.batch([float_image, read_input.label],
                                                          batch_size=batch_size,
                                                          num_threads=16,
                                                          capacity=min_queue_examples + 3 * batch_size
                                                          )
        return image_test, tf.reshape(label_test, [batch_size])


#-----------------------------------------------------------------------------------------------------------------------------#
import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval=10000
data_dir = 'E:/YAN/HelloWorld/cv/【11】CNN/cifar/cifar/cifar/cifar_data/cifar-10-batches-bin'

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    # tf.truncated_normal(shape,stddev=stddev):截断的产生正态分布的随机数，即随机数与均值的差值若大于两倍的标准差，则重新生成
    var = tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        # tf.nn.l2_loss()：一般用于优化目标函数中的正则项，防止参数太多复杂容易过拟合
        weights_loss = tf.multiply(tf.nn.l2_loss(var),w1,name='weights_loss')
        tf.add_to_collection('losses',weights_loss)
    return var

#使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
#其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
image_train,label_train = Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=True)
image_test,label_test = Cifar10_data.inputs(data_dir=data_dir,batch_size=batch_size,distorted=False)

#创建占位符
x = tf.placeholder(tf.float32,[batch_size,24,24,3])
y = tf.placeholder(tf.int64,[batch_size])


#第一个卷积层
kernel1 = variable_with_weight_loss([5,5,3,64],stddev=5e-2,w1=0.0)
"""tf.nn.conv2d():卷积函数
   [1,1,1,1]:四个方向的padding值，一般来说都是一样的
   padding="SAME"：输入输出尺寸不变
   tf.nn.bias_add(conv1,bias1)：conv1的结果加上bias1
"""
conv1 = tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
bias1 = tf.Variable(tf.constant(0.0,shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1,bias1))
pool1 = tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')


#第二个卷积层
kernel2 = variable_with_weight_loss([5,5,64,64],stddev=5e-2,w1=0.0)
conv2 = tf.nn.conv2d(pool1,kernel2,[1,1,1,1],padding='SAME')
bias2 = tf.Variable(tf.constant(0.1,shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2,bias2))
pool2 = tf.nn.max_pool(relu2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

#因为要进行全连接层的操作，所以这里使用tf.reshape()函数将pool2输出变成一维向量，因为FC只能接受一维的，并使用get_shape()函数获取扁平化之后的长度
reshape = tf.reshape(pool2,[batch_size,-1])
#get_shape()[1].value表示获取reshape之后的第二个维度的值
dim = reshape.get_shape()[1].value

#第一个全连接层
"""
tf.matmul()与tf.multiply()区别
tf.matmul()：将矩阵a乘以矩阵b，生成a*b
tf.multiply()：两个矩阵中对应元素各自相乘。实现的是元素级别的相乘，不是矩阵乘法。两个相乘的数必须有相同的数据类型
"""
weight1 = variable_with_weight_loss(shape=[dim,384],stddev=0.04,w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1,shape=[384]))
fc_1=tf.nn.relu(tf.matmul(reshape,weight1)+fc_bias1)

#第二个全连接层
weight2 = variable_with_weight_loss([384,192],stddev=0.04,w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1,shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1,weight2)+fc_bias2)

#最后一个全连接层，所以激活函数要用softmax
weight3 = variable_with_weight_loss([192,10],stddev=1 / 192.0,w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1,shape=[10]))
result = tf.add(tf.matmul(fc_2,weight3),fc_bias3)

#计算损失，包括权重参数的正则化损失和交叉熵损失。使用softmax回归之后的交叉熵
cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,labels=tf.cast(y,tf.int64))

#好处：把loss跟每个权重绑定的更加密切。原始的cross_entropy是只跟最终的结果有关，和标签做一个loss
#坏处：容易过拟合。很多情况下，并不会引起质变。加不加，效果其实没有那么明显。所以一般来讲，不用再加上每个权重loss
#tf.add_n():一个列表的元素相加
weights_with_l2_loss=tf.add_n(tf.get_collection('losses'))
#对于回归问题：解决的是对具体数值的预测。
#解决回归问题的神经网络一般只有一个输出节点，这个节点的输出值是预测值。对于回归问题，最常用的损失函数是均方误差。
loss = tf.reduce_mean(cross_loss)+weights_with_l2_loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

#函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
train_top_k = tf.nn.in_top_k(result,y,1)

init_op=tf.global_variables_initializer()

#运行
with tf.Session() as sess:
    sess.run(init_op)
    #启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作。
    #可以同时做16个batch
    coord = tf.train.Coordinator()
    tf.train.start_queue_runners(coord=coord)

    #每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    # #max_steps=epoch
    for step in range(max_steps):
        start_time = time.time()  # 获取当前时间
        image_batch,label_batch = sess.run([image_train,label_train])
        _,loss_value = sess.run([train_op,loss],feed_dict={x:image_batch,y:label_batch}) #因为train_op是优化器，没有返回值，所以sess.run后为None
        duration = time.time()-start_time
        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch"%(step,loss_value,examples_per_sec,sec_per_batch))

    #进行测试 计算最终的正确率
    num_batch = int(math.ceil((num_examples_for_eval/batch_size)))
    correct_num = 0
    total_num = num_batch * batch_size

    #在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([image_test, label_test])
        predictions = sess.run([train_top_k],feed_dict={x:image_batch,y:label_batch})
        correct_num += np.sum(predictions)
    #打印正确率信息
    print("accuracy = %.3f%%"%((correct_num/total_num) * 100))
