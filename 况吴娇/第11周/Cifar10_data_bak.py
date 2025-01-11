#该文件负责读取Cifar-10数据并对其进行数据增强预处理
import os ##导入Python的os模块，用于操作系统相关的功能，如文件路径操作。
import tensorflow as tf
num_classes=10  #定义变量num_classes，表示CIFAR-10数据集的类别数为10。

#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000 #表示每个训练周期的样本总数为50000
num_examples_pre_epoch_for_eval=10000 #表示每个评估周期的样本总数为10000

#定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass


#定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_queue): #定义函数read_cifar10，输入参数为file_queue（文件队列）。
    result=CIFAR10Record() #创建一个CIFAR10Record类的实例result，用于存储读取的数据。

    label_bytes=1                                            #如果是Cifar-100数据集，则此处为2
    result.height=32
    result.width=32
    result.depth=3                                           #因为是RGB三通道，所以深度是3

    image_bytes=result.height * result.width * result.depth  #图片样本总元素数量
    record_bytes=label_bytes + image_bytes                   #因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    reader=tf.FixedLengthRecordReader(record_bytes=record_bytes)  #使用tf.FixedLengthRecordReader()创建一个文件读取类。该类的目的就是读取文件
    #创建一个tf.FixedLengthRecordReader对象，用于读取固定长度的记录。
    result.key,value=reader.read(file_queue)                 #使用该类的read()函数从文件队列里面读取文件;用reader的read方法从文件队列中读取文件，返回键值对(key, value)。

    record_bytes=tf.decode_raw(value,tf.uint8)               #读取到文件以后，将读取到的文件内容从字符串形式解析为图像对应的像素数组
    #tf.decode_raw函数将读取到的文件内容解析为uint8类型的张量。
    #因为该数组第一个元素是标签，所以我们使用strided_slice()函数将标签提取出来，并且使用tf.cast()函数将这一个标签转换成int32的数值形式
    result.label=tf.cast(tf.strided_slice(record_bytes,[0],[label_bytes]),tf.int32)
    # 剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
    # 这一步是将一维数据转换成3维数据
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    '''
    tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes])：从解析后的张量中提取图像数据。
    tf.reshape(..., [result.depth, result.height, result.width])：将提取的图像数据重塑为三维数组，形状为(depth, height, width)。
    depth_major = ...：将重塑后的图像数据赋值给变量depth_major。
    '''
    # 我们要将之前分割好的图片数据使用tf.transpose()函数转换成为高度信息、宽度信息、深度信息这样的顺序
    # 这一步是转换数据排布方式，变为(h,w,c)
    result.uint8image = tf.transpose(depth_major, [1, 2, 0]) ##result.uint8image = ...：将调整后的图像数据赋值给result对象的uint8image属性。
    '''
    tf.transpose(a, perm)：这个函数用于重新排列输入张量 a 的维度。perm 参数是一个整数列表，指定了新的维度顺序。
    在CIFAR-10数据集中，图像数据最初是以 (depth, height, width) 的格式存储的，因为每个像素点的三个通道（RGB）是连续存储的。
    depth_major 变量的形状是 (3, 32, 32)，表示3个通道、32个高度和32个宽度。
    [1, 2, 0]：这个列表指定了新的维度顺序：
   1 表示原始的第二个维度（高度）成为新的第一个维度。
   2 表示原始的第三个维度（宽度）成为新的第二个维度。
   0 表示原始的第一个维度（深度）成为新的第三个维度。
    '''
    return result  # 返回值是已经把目标文件里面的信息都读取出来
'''
tf.strided_slice(record_bytes, [0], [label_bytes])：使用tf.strided_slice函数从解析后的张量中提取标签。
tf.cast(..., tf.int32)：将提取的标签转换为int32类型。
result.label = ...：将转换后的标签赋值给result对象的label属性。

tf.strided_slice(input, begin, end)：这个函数用于从输入张量中提取一个切片。begin 和 end 参数分别指定了切片的起始和结束位置。
input：要切片的输入张量，这里是 record_bytes。
begin：切片的起始位置，这里是 [0]，表示从第一个字节开始。
end：切片的结束位置，这里是 [label_bytes]，表示切片到 label_bytes 个字节的位置。对于CIFAR-10数据集，label_bytes 为1，因此这个切片操作提取的是第一个字节，即标签。
'''
'''
剩下的元素再分割出来，这些就是图片数据，因为这些数据在数据集里面存储的形式是depth * height * width，我们要把这种格式转换成[depth,height,width]
这一步是将一维数据转换成3维数据
'''
'''
def inputs(data_dir, batch_size, distorted)：定义函数inputs，输入参数为data_dir（数据目录），batch_size（批次大小），distorted（是否进行数据增强）。
filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]：使用列表推导式生成CIFAR-10数据集的文件路径列表。
'''


def inputs(data_dir, batch_size, distorted):  # 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]  # 拼接地址

    file_queue = tf.train.string_input_producer(filenames)  # 根据已经有的文件地址创建一个文件队列
    #使用tf.train.string_input_producer函数将文件路径列表转换为文件队列。
    read_input = read_cifar10(file_queue)  # 根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch = num_examples_pre_epoch_for_train
    '''
    num_examples_per_epoch = num_examples_pre_epoch_for_train：设置每个周期的样本总数为训练样本总数。
    '''

    if distorted != None:  # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])  # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        '''
        tf.random_crop(reshaped_image, [24, 24, 3])：
       目的：从图像中随机裁剪出一个固定大小的区域。
       作用：reshaped_image是输入图像，[24, 24, 3]是裁剪后的图像大小。这个操作用于数据增强，通过随机裁剪增加数据多样性。
       输出：cropped_image是裁剪后的图像。
       '''

        flipped_image = tf.image.random_flip_left_right(
            cropped_image)  # 将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数

        adjusted_brightness = tf.image.random_brightness(flipped_image,
                                                         max_delta=0.8)  # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数

        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2,
                                                     upper=1.8)  # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        '''
        目的：随机调整图像对比度。
        作用：在[lower, upper]范围内随机调整adjusted_brightness的对比度。lower=0.2和upper=1.8表示对比度调整的最小和最大范围。
        输出：adjusted_contrast是对比度调整后的图像。
        '''

        float_image = tf.image.per_image_standardization(
            adjusted_contrast)  # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        '''
        tf.image.per_image_standardization(adjusted_contrast)：
目的：对图像进行标准化处理。
作用：对每个像素减去平均值并除以标准差，使图像具有零均值和单位方差。
输出：float_image是标准化后的图像。
        '''
        float_image.set_shape([24, 24, 3])  # 设置图片数据及标签的形状
        read_input.label.set_shape([1]) #显式设置read_input.label的形状为(1,)，确保标签形状的一致性。

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)
        #作用：num_examples_pre_epoch_for_eval是每个评估周期的样本总数，乘以0.4得到队列中需要填充的最小样本数。
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        # 使用tf.train.shuffle_batch()函数随机产生一个batch的image和label
        # tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size, ...)：
        # 目的：从队列中随机抽取一个批次的图像和标签。
        # 作用：[float_image, read_input.label]是输入的图像和标签，batch_size是批次大小，num_threads是线程数，capacity是队列容量，min_after_dequeue是队列中最小样本数。
        # 输出：images_train和labels_train是训练用的图像和标签批次。


        return images_train, tf.reshape(labels_train, [batch_size])
       #目的：返回训练用的图像和标签批次。
       # 作用：tf.reshape(labels_train, [batch_size])将标签批次的形状调整为(batch_size,)。输出：images_train和labels_train是训练用的图像和标签批次。

    else:  # 不对图像数据进行数据增强处理
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24,
                                                               24)  # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切,将图像裁剪或填充到指定大小

        float_image = tf.image.per_image_standardization(resized_image)  # 剪切完成以后，直接进行图片标准化操作

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [batch_size])

'''tf.train.shuffle_batch 和 tf.train.batch 是 TensorFlow 中用于从队列中批量提取数据的两个不同函数。它们的主要区别在于数据是否被打乱：

tf.train.shuffle_batch
目的：从输入队列中随机抽取一个批次的数据。
特点：数据被打乱。这个函数会在队列中随机抽取样本，以确保每个批次的数据是随机的。这对于训练过程中的数据增强和防止过拟合非常有用。
使用场景：通常用于训练阶段，因为随机性有助于模型学习到更一般化的特征。
tf.train.batch
目的：从输入队列中按顺序提取一个批次的数据。
特点：数据不被打乱。这个函数会按顺序从队列中提取样本，保持数据的原始顺序。
使用场景：通常用于评估阶段，因为评估时通常需要按顺序处理数据，以便于比较模型的预测结果和真实标签。
共同点
批量处理：两者都用于批量处理数据，返回一个批次的样本和标签。
队列机制：两者都依赖于 TensorFlow 的队列机制来管理数据流。
选择使用哪个函数
训练时：通常使用 tf.train.shuffle_batch，因为训练时需要数据的随机性来增强模型的泛化能力。

'''