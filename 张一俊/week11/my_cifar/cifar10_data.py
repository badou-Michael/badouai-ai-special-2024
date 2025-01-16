# 本文件描述cifar数据读取过程
import os
import tensorflow as tf


# 常量定义
NUM_CLASEES = 10  # Cifar-10包含10个类别
NUM_TRAIN_EXAMPLES = 50000  # 用于训练的样本数
NUM_EVAL_EXAMPLES = 10000  # 用于评估的样本数

# 创建Cifar10数据样本类
class Cifar10Sample(object):
    pass

# 读取Cifar10数据集的文件
def read_cifar10(file_paths):
    sample = Cifar10Sample()

    # Cifar-10文件的格式
    label_bytes = 1  # 每个数据样本包含一个标间，标签字节数为1
    sample.height = 32  # 图像高32像素
    sample.width = 32  # 图像宽32像素
    sample.depth = 3  # 图像通道数是3(RGB)

    image_bytes = sample.height * sample.width * sample.depth   # 图像数据的字节数
    record_bytes = label_bytes + image_bytes  # 每个记录包含标签和图像数据

    # 开始读取数据
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)  # 创建一个FixedLengthRecordReader对象(固定长度记录的文件读取器)，送入record_bytes数据
    sample.key, value = reader.read(file_paths)  # 使用read方法读取：从输入文件队列 file_paths 中读取一条记录。 sample.key：记录的文件名、偏移量。value:读取到的记录的内容
                                                # file_paths 通常是通过 tf.train.string_input_producer 生成的一个文件路径队列，用于支持从多个文件中连续读取数据。
    # 分析数据
    raw_data = tf.io.decode_raw(value, tf.uint8)   # 解码读取的内容：将图像数据从字符串格式解析为原始的字节数据

    sample.label = tf.cast(tf.strided_slice(raw_data, [0], [label_bytes]), tf.int32)  # 提取标签，并将其转为tf.int32类型。从raw_data中提取[0,label_bytes)的内容

    # print(sample.label)

    image_data_depth_major = tf.reshape(tf.strided_slice(raw_data, [label_bytes], [label_bytes + image_bytes]), [sample.depth, sample.height, sample.width])  # 将提取的一维向量重新形状化为三维张量（深度、高度、宽度）——输入是 深度优先（Depth-Major） 格式,

    # 如果要转换数据的格式：
    image_data_height_major = tf.transpose(image_data_depth_major, [1, 2, 0])  # 假如要从深度优先的格式（depth, height, width）转为宽度优先（height, width, depth）

    # 存到sample中
    sample.uint8image = image_data_height_major

    # 返回包含标签和图像的对象
    return sample


# 预处理图像（数据增强与标准化）
def preprocess_image(image_bytes, apply_augmentation=None):
    # 转为float32
    image_float32 = tf.cast(image_bytes, tf.float32)  # 将图像数据转换为float32类型

    #　图像增强
    if apply_augmentation:
        #　如果需要图像增强，进行以下操作
        image_float32 = tf.image.random_crop(image_float32, [24, 24, 3])  # 随机裁剪图像到24 * 24              # 和 tf.image.resize(image, [32, 32])有啥区别
        image_float32 = tf.image.random_flip_left_right(image_float32)  # 随机左右翻转图像
        image_float32 = tf.image.random_brightness(image_float32, max_delta=0.8)  # 随机调整亮度
        image_float32 = tf.image.random_contrast(image_float32, lower=0.2, upper=1.8)  # 随机调整对比度
        # image = tf.image.random_flip_up_down(image_float32)  # 旋转操作

    # 标准化：对每个图像减去均值并除以标准差
    image_float32 = tf.image.per_image_standardization(image_float32)                               # 和tf.cast(image, tf.float32) / 255.0  # 转为float并归一化有啥区别
    return image_float32


# 输入函数，加载数据并进行预处理
def inputs(data_dir,  batch_size, apply_augmentation=None):
    # 文件路径拼接：Cifar-10包含5个数据批次
    file_paths = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]

    # 创建文件队列，加载所有数据并打乱顺序
    filename_queue = tf.train.string_input_producer(file_paths, shuffle=True)  # 这里不填shuffle参数的话，可以把tf.train.batch替换为tf.train.shuffle_batch实现打乱

    # 读取数据
    sample = read_cifar10(filename_queue)
    image = sample.uint8image
    label = sample.label

    # 对每个样本进行预处理
    image = preprocess_image(image, apply_augmentation)

    # 将标签从标量转换为[1]的形状
    label = tf.reshape(label, [1])  # 也可以使用label.set_shape([1])来实现

    # 创建批次（batching）
    # images, labels = tf.train.batch([image, label], batch_size=batch_size, num_threads=4, capacity=1000)
    images, labels = tf.train.shuffle_batch([image, label], batch_size=batch_size, num_threads=4, capacity=int(NUM_EVAL_EXAMPLES * 0.4) + 3 * batch_size, min_after_dequeue=int(NUM_EVAL_EXAMPLES * 0.4))
    # images_train, labels_train = tf.train.shuffle_batch([float_image, read_input.label], batch_size=batch_size, num_threads=16, capacity=min_queue_examples + 3 * batch_size, min_after_dequeue=min_queue_examples, )

    return images, labels


# # ############################## 调用案例 ############################## #
import tensorflow as tf
print(tf.__version__)




# tf.image.random_crop 和 tf.image.resize
# tf.image.random_crop 是随机裁剪图像到指定尺寸。
# tf.image.resize 是调整图像的尺寸，可能导致形变或插值。

# tf.image.per_image_standardization 和简单归一化 image / 255.0
# 前者是对每张图片减去均值并除以标准差，用于归一化像素值和提升训练稳定性。
# 后者是将像素值缩放到 [0, 1] 范围，适合简单的预处理。