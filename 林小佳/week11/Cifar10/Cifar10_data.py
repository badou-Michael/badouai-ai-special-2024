import os
import tensorflow as tf

num_classes = 10  # 分类类别个数

num_examples_pre_epoch_for_train = 50000    # 用于训练的样本数
num_examples_pre_epoch_for_eval = 10000     # 用于推理的样本数

# 定义一个用来返回 读取cifar10数据集 的空类
class CIFAR10Record(object):
    pass  # 空类，用作数据结构的占位符


# 读取函数
def read_cifar10(file_queue):
    result = CIFAR10Record()  # 实例化，将读取的cifar10数据集存到result里

    label_bytes = 1  # 指明要读取的数据集是cifar10
    result.height = 32
    result.width = 32
    result.depth = 3

    image_bytes = result.height * result.width * result.depth  # 算出图像的总像素点数，这就是样本总数
    record_bytes = label_bytes + image_bytes  # 要记录的数据=图像样本总数+对应标签值

    reader = tf.FixedLengthRecordReader(
        record_bytes=record_bytes)  # 使用tf.FixedLengthRecordReader()创建一个文件读取类，固定长度为record_bytes
    result.key, value = reader.read(file_queue)  # 调用文件读取类的read方法从指定的文件队列里读取文件内容
    record_bytes = tf.decode_raw(value, tf.uint8)  # 对读入的数据解码：将字节数据解码为无符号8位整型数组(也就是像素数组)

    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)  # 获取图像标签
    # tf.strided_slice 用于从数组中切片；tf.cast 用于类型转换

    # 将读取的一维图像数据重塑成三维形式：(depth * height * width)→(depth，height，width)
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])

    # 使用tf.transpose()将通道顺序由CHW转换成HWC
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])

    return result  # 返回包含标签和图像数据的CIFAR10Record实例


def inputs(data_dir, batch_size, distorted):  # inputs()函数是对输入的图像数据进行图像增强的预处理——提高训练数据集的质量和数量
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]  # 拼接生成文件地址

    file_queue = tf.train.string_input_producer(filenames)  # 由生成的文件地址创建文件队列
    read_input = read_cifar10(file_queue)  # 使用前面定义好的文件读取函数read_cifar10()来读取队列中的文件

    reshaped_image = tf.cast(read_input.uint8image, tf.float32)  # 将读取到的图片数据转成float32格式

    num_examples_per_epoch = num_examples_pre_epoch_for_train  # 设置每个周期的训练样本数

    if distorted != None:  # 若distorted参数不为空值，就代表要进行图像增强的预处理
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])  # tf.random_crop()函数 对图像进行剪切
        flipped_image = tf.image.random_flip_left_right(cropped_image)  # 将剪切后的图像进行左右翻转操作
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)  # 随机亮度调整操作
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)  # 随机对比度调整操作

        float_image = tf.image.per_image_standardization(adjusted_contrast)  # 对图像进行标准化操作：对每一个像素减去平均值并除以像素方差

        float_image.set_shape([24, 24, 3])  # 设置图像数据形状
        read_input.label.set_shape([1])  # 设置图像标签形状

        min_queue_examples = int(num_examples_pre_epoch_for_eval * 0.4)  # 计算最小队列样本数
        print("Filling queue with %d CIFAR images before starting to train.    This will take a few minutes."
              % min_queue_examples)

        images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size,
                                                            num_threads=16,
                                                            capacity=min_queue_examples + 3 * batch_size,
                                                            min_after_dequeue=min_queue_examples,
                                                            )
        # 使用tf.train.shuffle_batch()函数打乱数据，使得每个batch的image和label都是随机取得的

        return images_train, tf.reshape(labels_train, [batch_size])  # 返回训练图像和标签

    else:  # 不对图像数据进行图像增强预处理
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)
        # 函数tf.image.resize_image_with_crop_or_pad() 对图片数据进行剪切，使得图像大小满足需要

        float_image = tf.image.per_image_standardization(resized_image)  # 标准化操作

        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])

        min_queue_examples = int(num_examples_per_epoch * 0.4)

        images_test, labels_test = tf.train.batch([float_image, read_input.label],
                                                  batch_size=batch_size, num_threads=16,
                                                  capacity=min_queue_examples + 3 * batch_size)
        # 这里使用batch()函数代替tf.train.shuffle_batch()函数

        return images_test, tf.reshape(labels_test, [batch_size])
