# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
# 设置训练和评估的样本数量
num_for_train = 50000
num_for_eval = 10000

#定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10_record_self(object):
    pass

def read_cifar10_self(file_queue):
    # 定义一个函数，用于读取cifar10数据
    result =  CIFAR10_record_self()

    label_bytes = 1  # 如果是Cifar-100数据集，则此处为2
    # 数据集的大小
    result.height = 32
    result.width = 32
    result.depth = 3

    # 计算图片样本的总元素量
    image_bytes = result.height*result.width*result.depth

    # 一个样本包含一张图片和一个标签，最终元素量
    record_bytes = image_bytes + label_bytes

    # 创建一个文件读取类
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)

    # 读取到文件后，将读取内容从字符串转为uint8，及图片像素数组格式
    record_bytes = tf.decode_raw(value, tf.uint8)

    # 将标签分离出来，再转成int32格式：从第【0】位取【label_bytes】长度的元素
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

    # 将剩下的元素取出来，并转换为图像格式
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])

    # chw转hwc，tensorflow支持两种格式的数据，所以可以转也可以不转
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result

import os

def inputs_self(data_dir,batch_size,distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]

    file_queue = tf.train.string_input_producer(filenames)  # 根据已经有的文件地址创建一个文件队列
    read_input = read_cifar10_self(file_queue)  # 根据已经有的文件队列使用已经定义好的文件读取函数read_cifar10()读取队列中的文件

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将已经转换好的图片数据再次转换为float32的形式

    num_examples_per_epoch = num_for_train

    # 判断是否进行图像增强
    # 图像增强表示的是，在原始图像的基础上，对数据进行一定的改变，增加了数据样本的数量，但是
    # 数据的标签值并不发生改变。
    if distorted != None:
        # 随机裁切
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])

        # 左右翻转
        flipped_image = tf.image.random_flip_left_right(
            cropped_image)

        # 上下翻转
        flipped_image2 = tf.image.random_flip_up_down(flipped_image)

        # 亮度调整
        adjusted_brightness = tf.image.random_brightness(flipped_image2,
                                                         max_delta=0.5)
        # 颜色色调
        hue_image = tf.image.random_hue(adjusted_brightness, 0.5)

        # 调整对比度
        adjusted_contrast = tf.image.random_contrast(hue_image, lower=0.8,
                                                     upper=1.2)

        # 图像标准化，tf.image.per_image_standardization()函数对每一个像素减去平均值并除以像素方差
        float_image = tf.image.per_image_standardization(adjusted_contrast)


        float_image.set_shape([24, 24, 3])  # 设置图片数据及标签的形状
        read_input.label.set_shape([1])


        min_queue_examples = int(num_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        # 随机产生一个batch的image和label
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )

        return images_train, tf.reshape(labels_train, [batch_size])

    else:  # 不对图像数据进行数据增强处理
        # 将图像裁切为模型输入尺寸
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)

        # 图像标准化
        float_image = tf.image.per_image_standardization(resized_image)

        # 设置图片数据及标签的形状
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数
        return images_test, tf.reshape(labels_test, [batch_size])
