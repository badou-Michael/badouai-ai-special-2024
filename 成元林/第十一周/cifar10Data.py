import os

import tensorflow as tf

# 设置测试样本与训练样本数
train_num = 50000
test_num = 10000


class Cifar10Record(object):
    pass


def readCifar10(datapath):
    """
    读取cifar10的数据
    @param datapath:
    @return:
    """
    result = Cifar10Record()
    # cifar10的字节数是1，cifar100的字节数是2
    result.label_bytes = 1
    result.height, result.width, result.c = 32, 32, 3  # cifar的图片大小是32*32*3
    # 所有字节数
    record_bytes = result.label_bytes + result.height * result.width * result.c

    # 创建文件读取类,读取bin文件队列的
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    # 读取文件
    result.key, result.value = reader.read(datapath)
    # 图像解析，字符串解析为对应的图像
    record_bytes = tf.decode_raw(result.value, tf.uint8)
    # 通过截取获取标签与图片,标签转为对应的int类型，方便计算
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [result.label_bytes]), tf.int32)
    # 截取得到图片的部分
    image = tf.strided_slice(record_bytes, [result.label_bytes],
                             [result.label_bytes + result.height * result.width * result.c])
    # 原始图片是一维的，且形式为c,h,w的形式
    imageUint8 = tf.reshape(image, [result.c, result.height, result.width])
    # [1, 2, 0]指定了原始维度数组中各个维度的新位置。
    result.imageUint8 = tf.transpose(imageUint8, [1, 2, 0])
    return result


def handleInput(dirPath, batchSize, isImageEhanmance):
    """
    数据预处理
    @param dirPath:文件夹地址
    @param batchSize: 一个批次数量
    @param IsImageEhanmance: 是否图片增强
    @return:
    """
    # 拼接地址
    filepath = [os.path.join(dirPath, "data_batch_%d.bin" % i) for i in range(1,6)]
    # 将字符串地址转为文件队列
    filequepath = tf.train.string_input_producer(filepath)
    # 读取cifar10的数据
    readResult = readCifar10(filequepath)
    # 读取里边的图片并转为float32，便于后边计算
    result_image = tf.cast(readResult.imageUint8, tf.float32)

    if isImageEhanmance == True:
        # 随机裁剪图片
        cropImage = tf.image.random_crop(result_image, size=[24, 24, 3])
        # 随机左右反转
        leftrightImage = tf.image.random_flip_left_right(cropImage)
        # 随机调整对比度
        contrastImage = tf.image.random_contrast(leftrightImage, lower=0.2, upper=1.8)
        # 随机调整亮度
        brightnessImage = tf.image.random_brightness(contrastImage, max_delta=0.8)

        # 标准化图片
        standardizationImage = tf.image.per_image_standardization(brightnessImage)
        # 设置形状
        standardizationImage.set_shape([24, 24, 3])
        readResult.label.set_shape([1])
        # 队列最少容量
        min_num_of_queue = int(train_num * 0.4)
        image_train_batch, label_train_batch = tf.train.shuffle_batch([standardizationImage, readResult.label],
                                                                      batch_size=batchSize, num_threads=16,
                                                                      capacity=min_num_of_queue + 3 * batchSize,
                                                                      min_after_dequeue=min_num_of_queue)

        return image_train_batch, tf.reshape(label_train_batch,[batchSize])
    else:
        resizeImage = tf.image.resize_image_with_crop_or_pad(result_image, target_height=24, target_width=24)
        standardizationImage = tf.image.per_image_standardization(resizeImage)
        standardizationImage.set_shape([24, 24, 3])
        readResult.label.set_shape([1])
        min_num_of_queue = int(test_num * 0.4)
        image_test_batch, label_test_batch = tf.train.batch([standardizationImage, readResult.label],
                                                            batch_size=batchSize, num_threads=16,
                                                            capacity=min_num_of_queue + 3 * batchSize)
        return image_test_batch, tf.reshape(label_test_batch,[batchSize])

