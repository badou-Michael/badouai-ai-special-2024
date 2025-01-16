# 文件名：11周作业_Cifar10.py
# 该模块构建神经网络的整体结构，并进行训练和测试（评估）过程

import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data1  # 导入数据处理模块

# 定义一些参数
max_steps = 4000  # 训练的最大步数
batch_size = 100
num_examples_for_eval = 10000  # 测试集样本数量
data_dir = 'Cifar_data/cifar-10-batches-bin'  # CIFAR-10 数据集目录

# 定义一个函数，用于创建带有L2正则化的变量
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var

# 获取训练和测试数据
images_train, labels_train = Cifar10_data1.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = Cifar10_data1.inputs(data_dir=data_dir, batch_size=batch_size, distorted=False)

# 定义输入数据的占位符
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_ = tf.placeholder(tf.int32, [batch_size])

# 构建第一个卷积层
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='SAME')

# 构建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1],
                       strides=[1, 2, 2, 1], padding='SAME')

# 扁平化处理
reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# 全连接层 1
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 全连接层 2
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc2 = tf.nn.relu(tf.matmul(fc1, weight2) + fc_bias2)

# 输出层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
logits = tf.add(tf.matmul(fc2, weight3), fc_bias3)

# 计算损失，包括交叉熵和正则化损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=tf.cast(y_, tf.int64))
cross_entropy_mean = tf.reduce_mean(cross_entropy)
tf.add_to_collection('losses', cross_entropy_mean)
loss = tf.add_n(tf.get_collection('losses'))

# 定义训练步骤
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

# 计算预测的正确率
top_k_op = tf.nn.in_top_k(logits, y_, 1)

# 初始化变量
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()

    # 训练过程
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("Step %d, loss=%.2f (%.1f examples/sec; %.3f sec/batch)" %
                  (step, loss_value, examples_per_sec, sec_per_batch))

    # 评估模型
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    true_count = 0
    total_sample_count = num_batch * batch_size

    for _ in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run(top_k_op, feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    precision = true_count / total_sample_count
    print("Accuracy = %.3f%%" % (precision * 100))
