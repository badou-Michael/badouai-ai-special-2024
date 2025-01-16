import os.path

import tensorflow as tf
from tensorboard.plugins.beholder.im_util import read_image

num_classes = 10

#
num_examples_per_epoch_for_train = 50000
num_examples_per_epoch_for_eval = 10000

class CIFAR10Record(object):
    pass

# 读取文件流中内容
def read_cifar10(file_queue):
    result = CIFAR10Record()

    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes

    # 创建tf文件读取类读取文件流
    reader = tf.FixedLengthRecordReader(record_bytes)
    result.key, value = reader.read(file_queue)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     sess.run(tf.local_variables_initializer())
    #     keys, values = sess.run([result.key, value])
    #     print("result.key:", keys)
    #     print("values", values)

    record_bytes = tf.decode_raw(value, tf.uint8) # 解析为像素数组
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32) # 解析出label
    # 分割出图片数据并reshape shape(d, h, w)
    depth_major_img = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes])
                                 , [result.depth, result.height, result.width])
    # shape(h, w, d)
    result.uint8img = tf.transpose(depth_major_img, [1, 2, 0])
    return result

def inputs(data_dir, batch_size, is_train):
    if is_train:
        file_names = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    else:
        file_names = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(5, 6)]
    file_queue = tf.train.string_input_producer(file_names)
    read_input = read_cifar10(file_queue)
    reshaped_img = tf.cast(read_input.uint8img, tf.float32)

    num_examples_per_epoch = num_examples_per_epoch_for_train
    resized_img = tf.image.resize_image_with_crop_or_pad(reshaped_img, 24, 24) # 图片裁剪
    float_img = tf.image.per_image_standardization(resized_img) # 图片标准化
    read_input.label.set_shape([1]) # 设为静态张量，固定形状

    min_queue_examples = int(num_examples_per_epoch * 0.4)

    image_test, label_test = tf.train.batch([float_img, read_input.label], batch_size=batch_size
                                            , num_threads=16
                                            , capacity=min_queue_examples + batch_size * 3)
    return image_test, tf.reshape(label_test, [batch_size])
