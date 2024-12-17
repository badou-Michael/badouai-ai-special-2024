import tensorflow.compat.v1 as tf
import os


tf.disable_v2_behavior()
#设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train=50000
num_examples_pre_epoch_for_eval=10000

class Cifar10_DataRecorder():
    pass

def read_cifar(file_queue):
    """
        定义一个函数读取目标文件里的内容
        输入：file_queue 二进制文件bin
        输出：一个类的对象，用于存储cifar-10数据
    """
    result = Cifar10_DataRecorder()
    result.width = 32
    result.height = 32
    result.depth = 3
    label = 1

    record_bytes = result.width * result.height * result.depth + label

    reader = tf.FixedLengthRecordReader(record_bytes) #FixedLengthRecordReader()是一个类，创建一个类的对象，用于固定长度的文件读取
    key, value = reader.read(file_queue)  #read()是FixedLengthRecordReader()的一个方法，用于从文件队列中读取文件

    img_value = tf.decode_raw(value, tf.uint8) #tf.decode_raw()用于将TFrecord的数据（字符串)解码为原来的数据格式
    result.label = tf.cast(tf.strided_slice(img_value, [0], [label]), tf.int32) #tf.cast用于强制类型转换， 提取标签
    major_data = tf.strided_slice(img_value, [label], [record_bytes]) #将除了标签之外的其余数据提取出来
    major_data = tf.reshape(major_data, [result.depth, result.height, result.width])  #将图片转化为[c, h, w]的格式

    result.uint8img = tf.transpose(major_data, [1,2,0]) #将数据从（c,h,w)格式转化为（h,w,c)

    return result

def data_inputs(data_dir, batch_size, distorted):
    """对数据进行处理，以及是否要进行数据增强进行判断
    输入： 1.data_dir: 读取数据集的文件路径
          2.batch_size
          3.distorted: 是否要进行数据增强的操作
    输出： 1.train : 训练集
          2.label: 标签
    """

    file_path = [os.path.join(data_dir, ('data_batch_%d.bin'%i)) for i in range(1,6)]
    file_queue = tf.train.string_input_producer(file_path)  #创建一个文件队列用于存储输入文件
    read_input = read_cifar(file_queue)
    reshaped_input = tf.cast(read_input.uint8img, tf.float32)
    
    # print(read_input.uint8img)
    # print(read_input.label)

    if distorted == None: #不进行数据增强

        resized_img = tf.image.resize_image_with_crop_or_pad(reshaped_input, 24, 24) #使用该函数将图片裁减为24*24的大小
        float_img = tf.image.per_image_standardization(resized_img)  #(24,24,3)
        # float_img.set_shape([24,24,3])
        read_input.label.set_shape([1])   #shape = [32,1]
        min_queue_examples=int(num_examples_pre_epoch_for_train * 0.4)

        image_train, label_train = tf.train.batch([float_img, read_input.label], batch_size, min_queue_examples, 1)
        # image_train = tf.train.batch([float_img], batch_size, min_queue_examples, 2)
        # label_train = tf.train.batch([read_input.label], batch_size, min_queue_examples, 2)
        return image_train, tf.reshape(label_train, [batch_size])
        # return label_train

    else: #如果进行数据增强
        """步骤：
        1.裁减 24*24
        2.翻转 random_flip_left_right()
        3.调整亮度 random_brightness()
        4.调整对比度 random_contrast()
        5.图像标准化 per_image_standardization()
        """
        cropped_img = tf.random_crop(reshaped_input, [24,24,3])
        flipped_img = tf.image.random_flip_left_right(cropped_img)
        brightness_img = tf.image.random_brightness(flipped_img, max_delta = 0.2)
        contrast_img = tf.image.random_contrast(brightness_img, 0.2, 0.5)
        standard_img = tf.image.per_image_standardization(contrast_img)

        read_input.label.set_shape([1])
        min_queue_examples=int(num_examples_pre_epoch_for_train * 0.4)
        image_train, label_train = tf.train.batch([standard_img, read_input.label], batch_size, min_queue_examples, 1)
        return image_train, tf.reshape(label_train, [batch_size])

    
# inputimage, inputlabel = data_inputs('/home/zzy/work/cifar/cifar_data/cifar-10-batches-bin', 32, True)
# print(inputimage)
# print(inputlabel)
# # with tf.Session() as sess:




