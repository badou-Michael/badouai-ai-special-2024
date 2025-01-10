import os

import tensorflow as tf

# 设定用于训练和评估的样本总数
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000


# 定义一个空类，用于返回读取的Cifar-10的数据
class cifar10_data(object):
    pass


# 定义一个读取Cifar-10的函数read_cifar10()，这个函数的目的就是读取目标文件里面的内容
def read_cifar10(file_path):
    label_bytes = 1  # 如果是Cifar-100数据集，则此处为2
    height = 32
    width = 32
    depth = 3  # 因为是RGB三通道，所以深度是3
    image_bytes = height * width * depth  # 图片样本总元素数量
    record_bytes = label_bytes + image_bytes  # 因为每一个样本包含图片和标签，所以最终的元素数量还需要图片样本数量加上一个标签值

    def parse_record(serialized_example):
        features = tf.io.parse_single_example(
            serialized_example,
            features={
                'label': tf.io.FixedLenFeature([], tf.int64),
                'image_raw': tf.io.FixedLenFeature([], tf.string),
            })
        image = tf.io.decode_raw(features['image_raw'], tf.uint8)
        label = tf.cast(features['label'], tf.int32)

        # 将一维数据转换成3维数据
        depth_major = tf.reshape(image, [depth, height, width])
        # 转换数据排布方式，变为(h,w,c)
        image = tf.transpose(depth_major, [1, 2, 0])
        return image, label

    dataset = tf.data.FixedLengthRecordDataset(file_path, record_bytes=record_bytes)
    dataset = dataset.map(parse_record)
    return dataset


# 这个函数就对数据进行预处理---对图像数据是否进行增强进行判断，并作出相应的操作
# distorted:扭曲
def pre_process_images(data_dir, batch_size, distorted):
    file_names = [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1, 6)]
    dataset = tf.data.Dataset.from_tensor_slices(file_names)
    dataset = dataset.interleave(lambda x: read_cifar10(x), cycle_length=5, block_length=1)
    dataset = dataset.map(lambda image, label: (tf.cast(image, tf.float32), label))

    num_examples_pre_epoch = num_examples_pre_epoch_for_train

    if distorted:
        # 如果预处理函数中的distorted参数不为空值，就代表要进行图片增强处理

        # 首先将预处理好的图片进行剪切，使用tf.random_crop()函数
        dataset = dataset.map(lambda image, label: (tf.image.random_crop(image, [24, 24, 3]), label))

        # 将剪切好的图片进行左右翻转，使用tf.image.random_flip_left_right()函数
        dataset = dataset.map(lambda image, label: (tf.image.random_flip_left_right(image), label))

        # 将左右翻转好的图片进行随机亮度调整，使用tf.image.random_brightness()函数
        dataset = dataset.map(lambda image, label: (tf.image.random_brightness(image, max_delta=63 / 255.0), label))

        # 将亮度调整好的图片进行随机对比度调整，使用tf.image.random_contrast()函数
        dataset = dataset.map(lambda image, label: (tf.image.random_contrast(image, lower=0.2, upper=1.8), label))

        # 进行标准化图片操作，tf.image.per_image_standardization()函数是对每一个像素减去平均值并除以像素方差
        dataset = dataset.map(lambda image, label: (tf.image.per_image_standardization(image), label))

        # 设置图片数据及标签的形状
        dataset = dataset.shuffle(buffer_size=int(num_examples_pre_epoch * 0.4))
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
    else:
        # 不对图像数据进行数据增强处理

        # 在这种情况下，使用函数tf.image.resize_image_with_crop_or_pad()对图片数据进行剪切
        dataset = dataset.map(lambda image, label: (tf.image.resize_with_crop_or_pad(image, 24, 24), label))

        # 剪切完成以后，直接进行图片标准化操作
        dataset = dataset.map(lambda image, label: (tf.image.per_image_standardization(image), label))

        # 设置图片数据及标签的形状
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        return dataset
