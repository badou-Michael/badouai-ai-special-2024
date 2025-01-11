import tensorflow as tf
import numpy as np
import time
import math
import os

# CIFAR-10 数据集参数
data_dir = "./cifar-10-batches-bin"
num_classes = 10                         # CIFAR-10 数据集中有 10 个类别
batch_size = 100                         # 每次训练迭代中使用的样本数量
max_steps = 4000                         # 训练的最大步数
num_examples_for_eval = 10000            # 测试集中的样本总数

# 定义一个空类 CIFAR10Record，用来保存从二进制文件中读取的数据记录
class CIFAR10Record(object):
    pass

# 从给定的文件队列中读取单个 CIFAR-10 样本，解析出标签和图像，并调整图像格式为 [height, width, depth]
def read_cifar10(file_queue):
    result = CIFAR10Record()
    label_bytes = 1
    result.height = 32
    result.width = 32
    result.depth = 3
    image_bytes = result.height * result.width * result.depth
    record_bytes = label_bytes + image_bytes
    reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
    result.key, value = reader.read(file_queue)
    record_bytes = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)
    depth_major = tf.reshape(tf.strided_slice(record_bytes, [label_bytes], [label_bytes + image_bytes]),
                             [result.depth, result.height, result.width])
    result.uint8image = tf.transpose(depth_major, [1, 2, 0])
    return result

def inputs(data_dir, batch_size, distorted=None):
    filenames = [os.path.join(data_dir, "data_batch_%d.bin" % i) for i in range(1, 6)]
    # 创建一个文件队列，读取文件名列表中的文件
    file_queue = tf.train.string_input_producer(filenames)
    # 从文件队列中读取数据，返回一个 CIFAR10Record 对象，包含图像和标签
    read_input = read_cifar10(file_queue)
    reshaped_image = tf.cast(read_input.uint8image, tf.float32)

    # 如果 distorted 为 True，则进行数据增强
    if distorted:
        cropped_image = tf.random_crop(reshaped_image, [24, 24, 3])                                # 随机裁剪图像
        flipped_image = tf.image.random_flip_left_right(cropped_image)                             # 随机水平翻转图像
        adjusted_brightness = tf.image.random_brightness(flipped_image, max_delta=0.8)             # 随机调整图像亮度
        adjusted_contrast = tf.image.random_contrast(adjusted_brightness, lower=0.2, upper=1.8)    # 随机调整图像对比度
        float_image = tf.image.per_image_standardization(adjusted_contrast)                        # 对图像进行标准化处理
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_for_eval * 0.4)
        print("Filling queue with %d CIFAR images before starting to train. This will take a few minutes." % min_queue_examples)
        images_train, labels_train = tf.train.shuffle_batch(
            [float_image, read_input.label],
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples
        )
        return images_train, tf.reshape(labels_train, [batch_size])
    else:
        # 不进行数据增强
        resized_image = tf.image.resize_image_with_crop_or_pad(reshaped_image, 24, 24)              # 裁剪或填充图像
        float_image = tf.image.per_image_standardization(resized_image)                             # 标准化处理
        float_image.set_shape([24, 24, 3])
        read_input.label.set_shape([1])
        min_queue_examples = int(num_examples_for_eval * 0.4)
        images_test, labels_test = tf.train.batch(
            [float_image, read_input.label],
            batch_size=batch_size,
            num_threads=16,
            capacity=min_queue_examples + 3 * batch_size
        )
        return images_test, tf.reshape(labels_test, [batch_size])

# 创建一个variable_with_weight_loss()函数
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # 得到一个形状为shape，标准差为stddev的正态分布随机变量
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")  # 计算变量的平方和再乘以w1
        tf.add_to_collection("losses", weights_loss)  # 将 weights_loss 添加到 losses 集合中
    return var

# 构建模型
images_train, labels_train = inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 第一层卷积层
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')  # 'SAME' 表示在输入数据的边缘填充 0，使得卷积操作后输出的特征图与输入数据的尺寸相同
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))  # 通过将 bias1 广播到 conv1 的每个通道上，然后将它们相加
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

# 第二层卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

reshape = tf.reshape(pool2, [batch_size, -1])  # 重塑 pool2 的形状
dim = reshape.get_shape()[1].value

# 第一层全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)  # 第一层 384 个神经元 
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 第二层全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)  # 第二层 192 个神经元
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 第三层全连接层
weight3 = variable_with_weight_loss(shape=[192, num_classes], stddev=1 / 192.0, w1=0.0)  # 第三层 10 个神经元
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)

# 计算交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # 使用Adam优化器，学习率为1e-3

top_k_op = tf.nn.in_top_k(result, y_, 1)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)                # 初始化所有 TensorFlow 变量
    tf.train.start_queue_runners()

    for step in range(max_steps):
        start_time = time.time()     # 记录当前时间
        image_batch, label_batch = sess.run([images_train, labels_train])  # 获取一个批次的图像和标签
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time
        # 每 100 步打印训练信息
        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)" %
                  (step, loss_value, examples_per_sec, sec_per_batch))

    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))