#-*- coding:utf-8 -*-
# author: 王博然
import os
import tensorflow as tf
num_classes = 10 # cifar10

# 共6w张, 5w用于训练, 1w用于测试
num_train = 50000
num_eval  = 10000

# 定义空类,返回Cifar10的数据
class CIFAR10Record:
    pass

# 从文件队列里读取数据
def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1 # 0~9, 如果是 cifar-100 则为 2
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    record_bytes = image_bytes + label_bytes
    reader = tf.FixedLengthRecordReader(record_bytes = record_bytes)
    key, value = reader.read(file_queue)
    raw_bytes = tf.decode_raw(value, tf.uint8) # 将读取到的文件内容从字符串形式解析为图像对应的像素数组

    result.label = tf.cast(tf.strided_slice(raw_bytes,[0],[label_bytes]), tf.int32)
    # 一维数据转三维
    depth_major = tf.reshape(tf.strided_slice(raw_bytes, [label_bytes], [label_bytes + image_bytes]),
                            [result.depth, result.height, result.width])
    # 分割好的图片数据转换成为高度信息、宽度信息、深度信息这样的顺序(h,w,c)
    result.uint8image = tf.transpose(depth_major, [1,2,0])
    return result

# 对外方法
def inputs(data_dir, batch_size, distorted):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1,6)]
    file_queue = tf.train.string_input_producer(filenames)
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32) # 将已经转换好的图片数据再次转换为float32的形式

    if distorted != None:  # 图片增强
        cropped_image = tf.random_crop(reshaped_image, [24,24,3])
        flipped_image = tf.image.random_flip_left_right(cropped_image)
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta = 0.8) # 随机亮度调整
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8) # 对比度调整
        float_image = tf.image.per_image_standardization(adjusted_contrast)  # 进行标准化

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_eval * 0.4)
        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label],
                                                            batch_size = batch_size, num_threads = 16,
                                                            capacity = min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue = min_queue_examples)
        return images_train, tf.reshape(labels_train, [batch_size])
    else:                  # 不对图片进行增强处理
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24) # 图片裁切
        float_image = tf.image.per_image_standardization(resized_image) # 标准化

        # Tensor.set_shape 只是静态地为张量设置形状信息，不会影响张量的数据内容
        # 如果输入张量的形状不确定，可以使用此方法帮助推断并确保图执行时不会出错
        # 比如，在使用占位符（placeholder）时，我们通常会用 set_shape 明确指定占位符的形状。
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_train * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label], # label在 read_cifar10时赋值
                                                  batch_size = batch_size, num_threads = 16,
                                                  capacity = min_queue_examples + 3 * batch_size)
        return images_test, tf.reshape(labels_test, [batch_size])