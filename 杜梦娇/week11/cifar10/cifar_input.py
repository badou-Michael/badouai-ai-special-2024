import os
import tensorflow as tf

num_classes = 10
train_data_num = 50000
eval_date_num = 10000

# 定义一个空类，用于返回读取的Cifar-10的数据
class CIFAR10Record(object):
    pass

# 读取数据（图像数据和标签）
def read_data(file):
    Data = CIFAR10Record()
    Data.h = 32
    Data.w = 32
    Data.c = 3
    label_bytes = 1
    # 计算图像数据总像素个数
    images_bytes = Data.h * Data.w * Data.c
    # 计算图像数据总像素个数+标签（数据总元素个数）
    images_bytes_total = images_bytes + label_bytes

    # 创建 FixedLengthRecordReader
    reader = tf.FixedLengthRecordReader(record_bytes=images_bytes_total)
    # 读取文件
    Data.key, value = reader.read(file)
    # 解码读取的数据（字符串->数组（像素））
    images_bytes_total = tf.decode_raw(value, tf.uint8)
    # 获取数据标签
    Data.label = tf.cast(tf.strided_slice(images_bytes_total, [0], [label_bytes]), tf.int32)
    # 获取图像数据（c*h*w）pytorch支持h,w,c; t两种都支持
    images = tf.reshape(tf.strided_slice(images_bytes_total, [label_bytes], [label_bytes + images_bytes]), [Data.c, Data.h, Data.w])
    # 这一步是转换数据排布方式，变为(h,w,c)
    Data.uint8image = tf.transpose(images,[1,2,0])
    return Data

# 定义数据预处理
def input_data(data_dir, batch_size, distirted):
    # 读取批处理文件
    file = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    file_queue = tf.train.string_input_producer(file)
    input_data = read_data(file_queue)
    # 数据类型转换
    images_float= tf.cast(input_data.uint8image, tf.float32)

    #数据增强
    if distirted != None:
        images_cropped = tf.random_crop(images_float, [28, 28, 3])
        images_flipped = tf.image.random_flip_left_right(images_cropped)#裁剪图像，随机选择一个区域进行裁剪；tf.image.crop_to_bounding_box是根据指定的坐标和大小进行裁剪
        images_brightness = tf.image.random_brightness(images_flipped, max_delta=0.8)# 调节图像亮度
        # images_ = tf.image.random_saturation(images_brightness, lower=0.5, upper=1.5)# 随机调整图像的饱和度
        # images = tf.image.random_hue(images_, max_delta=0.2)# 随机调整图像的色相
        images_contrasted = tf.image.random_contrast(images_brightness, lower=0.5, upper=1.5)
        images = tf.image.per_image_standardization(images_contrasted)# 标准化图像：对每个像素减均值除方差；对图像进行线性缩放，使其均值为 0，标准差为 1，

        # 设置图片数据及标签的形状
        images.set_shape([28,28,3])
        input_data.label.set_shape([1])

        min_queue_examples = int(eval_date_num * 0.4) # 最小队列大小
        #将数据划分成不同的batch,从输入队列中随机抽取批次样本。它通常用于训练过程中，以便在每个批次中随机化数据，这有助于模型的泛化能力。
        images_train, label_train = tf.train.shuffle_batch([images, input_data.label], batch_size=batch_size, num_threads=8, capacity=min_queue_examples + 3 * batch_size,min_after_dequeue=min_queue_examples)
        return images_train, tf.reshape(label_train, [batch_size])

    else:
        images_resized = tf.image.resize_image_with_crop_or_pad(images_float, 28, 28) # 裁剪图片
        images = tf.image.per_image_standardization(images_resized)# 标准化图像：对每个像素减均值除方差；对图像进行线性缩放，使其均值为 0，标准差为 1，

        # 设置图片数据及标签的形状
        images.set_shape([28,28,3])
        input_data.label.set_shape([1])

        min_queue_examples = int(eval_date_num * 0.4) # 最小队列大小
        #将数据划分成不同的batch,从输入队列中随机抽取批次样本。它通常用于训练过程中，以便在每个批次中随机化数据，这有助于模型的泛化能力。
        images_train, label_train = tf.train.batch([images, input_data.label], batch_size=batch_size, num_threads=8, capacity=min_queue_examples + 3 * batch_size)
        return images_train, tf.reshape(label_train, [batch_size])
