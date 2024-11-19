# -*- coding: utf-8 -*-
# time: 2024/11/18 15:20
# file: Cifar10_data.py.py
# author: flame
import os

import tensorflow as tf

''' 定义 CIFAR-10 数据集的基本参数和记录类。 '''
''' 定义类别数量，CIFAR-10 数据集包含 10 个类别。 '''
num_classes = 10

''' 定义每个训练 epoch 的样本数，CIFAR-10 训练集包含 50000 个样本。 '''
num_examples_pre_epoch_for_train = 50000

''' 定义每个验证 epoch 的样本数，CIFAR-10 测试集包含 10000 个样本。 '''
num_examples_pre_epoch_for_eval = 10000

''' 定义一个简单的类来存储 CIFAR-10 记录的数据。 '''
class CIFAR10Record(object):
    ''' 该类用于存储从二进制文件中读取的 CIFAR-10 记录，包括图像和标签。 '''
    pass

''' 读取 CIFAR-10 数据集并进行预处理，支持图像增强和批量处理。 '''

''' 定义读取 CIFAR-10 数据集的函数，从文件队列中读取数据并解析成图像和标签。 '''
def read_cifar10(file_queue):
    ''' 创建一个 CIFAR10Record 对象来存储结果。 '''
    result = CIFAR10Record()
    ''' 定义标签字节数。 '''
    label_bytes = 1
    ''' 定义图像的高度。 '''
    result.height = 32
    ''' 定义图像的宽度。 '''
    result.width = 32
    ''' 定义图像的深度（通道数）。 '''
    result.depth = 3

    ''' 计算单个图像的字节数。 '''
    image_bytes = result.height * result.width * result.depth
    ''' 计算记录的总字节数（标签 + 图像）。 '''
    record_bytes = label_bytes + image_bytes
    ''' 创建一个 FixedLengthRecordReader 来读取固定长度的记录。 '''
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    ''' 从文件队列中读取一条记录。 '''
    result.key, value = reader.read(file_queue)
    ''' 解码记录为原始字节。 '''
    record_bytes = tf.decode_raw(value, tf.uint8)
    ''' 提取标签部分并转换为整型。 '''
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    ''' 提取图像部分并重塑为深度优先的张量。 '''
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]), (result.depth, result.height, result.width))
    ''' 将图像张量转换为高度、宽度、深度的格式。 '''
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    ''' 返回包含图像和标签的结果对象。 '''
    return result

''' 定义输入数据的处理函数，支持图像增强和批量处理。 '''
def inputs(data_dir, batch_size, distorted):
    ''' 构建数据文件的路径列表。 '''
    file_names = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    ''' 创建一个字符串输入生产者来管理文件队列。 '''
    file_queue = tf.train.string_input_producer(file_names)
    ''' 读取并解析 CIFAR-10 数据。 '''
    read_input = read_cifar10(file_queue)
    ''' 将图像数据转换为浮点类型。 '''
    reshape_image = tf.cast(read_input.uint8image, tf.float32)
    ''' 定义每个 epoch 的样本数。 '''
    num_epoch_per_epoch = num_examples_pre_epoch_for_train

    ''' 判断是否需要进行图像增强。 '''
    if distorted is not None:
        ''' 随机裁剪图像到 24x24 大小。 '''
        croped_image = tf.random_crop(reshape_image, [24, 24, 3])
        ''' 随机水平翻转图像以增加多样性。 '''
        fliped_image = tf.image.random_flip_left_right(croped_image)
        ''' 随机调整图像亮度，最大变化范围为 0.8。 '''
        adjusted_brightness = tf.image.random_brightness(fliped_image, max_delta=0.8)
        ''' 随机调整图像对比度，最大变化范围为 0.2 到 1.8。 '''
        adjused_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)
        ''' 对图像进行标准化处理，使其数据更稳定。 '''
        float_image = tf.image.per_image_standardization(adjused_contrast)
        ''' 设置图像数据的形状。 '''
        float_image.set_shape([24, 24, 3])
        ''' 设置标签的形状。 '''
        read_input.label.set_shape([1])
        ''' 计算最小队列示例数，用于后续的批处理。 '''
        min_queue_examples = int(num_epoch_per_epoch * 0.4)
        ''' 打印提示信息，告知用户正在填充队列。 '''
        print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.' % min_queue_examples)
        ''' 使用 tf.train.shuffle_batch 函数对图像进行批处理，并返回处理后的图像和标签。 '''
        image_train, label_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size, num_threads=16, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples)
        ''' 打印加载的图像和标签信息。 '''
        print(f"Loaded images and labels from {data_dir}")
        print(f"Images shape: {image_train.get_shape()}, Labels shape: {label_train.get_shape()}")
        ''' 返回处理后的图像和标签。 '''
        return image_train, tf.reshape(label_train, [batch_size])
    else:
        ''' 裁剪图像到 24x24 大小。 '''
        resized_image = tf.image.resize_image_with_crop_or_pad(reshape_image, 24, 24)
        ''' 对图像进行标准化处理，使其数据更稳定。 '''
        float_image = tf.image.per_image_standardization(resized_image)
        ''' 设置图像数据的形状。 '''
        float_image.set_shape([24, 24, 3])
        ''' 设置标签的形状。 '''
        read_input.label.set_shape([1])
        ''' 计算最小队列示例数，用于后续的批处理。 '''
        min_queue_examples = int(num_epoch_per_epoch * 0.4)
        ''' 使用 tf.train.batch 函数对图像进行批处理，并返回处理后的图像和标签。 '''
        images_test, labels_test = tf.train.batch([float_image, read_input.label], batch_size=batch_size, num_threads=16, capacity=min_queue_examples + 3 * batch_size)
        ''' 打印加载的图像和标签信息。 '''
        print(f"Loaded images and labels from {data_dir}")
        print(f"Images shape: {images_test.get_shape()}, Labels shape: {labels_test.get_shape()}")
        ''' 返回处理后的图像和标签。 '''
        return images_test, tf.reshape(labels_test, [batch_size])
