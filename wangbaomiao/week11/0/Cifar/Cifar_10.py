# -*- coding: utf-8 -*-
# time: 2024/11/18 18:14
# file: Cifar_10.py
# author: flame
import math

import numpy as np
import tensorflow as tf
import time
import Cifar10_data

''' 
此代码实现了一个基于卷积神经网络（CNN）的CIFAR-10图像分类模型。首先定义了训练参数和数据路径，然后构建了模型的各个层，包括卷积层、池化层和全连接层。接着定义了损失函数和优化器，并在训练过程中定期评估模型的精度。
'''

''' 定义训练的最大步数 '''
max_steps = 4000

''' 定义每批次的大小 '''
batch_size = 100

''' 定义用于评估的样本数量 '''
num_exaples_for_eval = 10000

''' 定义数据目录 '''
data_dir = "cifar_data/cifar-10-batches-bin"

''' 定义一个带有权重衰减的变量生成函数 '''
def variable_with_weight_loss(shape, stddev, w1):
    ''' 初始化变量，使用截断正态分布生成初始值 '''
    initial = tf.truncated_normal(shape, stddev=stddev)
    ''' 创建变量 '''
    var = tf.Variable(initial)
    ''' 如果 w1 不为 None，则计算 L2 正则化损失并添加到集合 'losses' 中 '''
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    ''' 返回创建的变量 '''
    return var

''' 加载训练数据 '''
images_train, labels_train = Cifar10_data.inputs(data_dir, batch_size, True)

''' 加载测试数据 '''
images_test, labels_test = Cifar10_data.inputs(data_dir, batch_size, None)

''' 定义输入占位符 x，形状为 [batch_size, 24, 24, 3]，表示批量大小、图像高度、宽度和通道数 '''
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])

''' 定义标签占位符 y，形状为 [batch_size]，表示批量大小 '''
y = tf.placeholder(tf.int32, [batch_size])

''' 定义第一个卷积层的权重，形状为 [5, 5, 3, 64]，标准差为 5e-2，不使用 L2 正则化 '''
kernel1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, w1=0.0)

''' 应用卷积操作，步长为 1，填充方式为 SAME '''
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding='SAME')

''' 定义第一个偏置项，形状为 [64]，初始值为 0.0 '''
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))

''' 应用 ReLU 激活函数 '''
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))

''' 应用最大池化操作，池化窗口大小为 3x3，步长为 2，填充方式为 SAME '''
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

''' 定义第二个卷积层的权重，形状为 [5, 5, 64, 64]，标准差为 5e-2，不使用 L2 正则化 '''
kernel2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, w1=0.0)

''' 应用卷积操作，步长为 1，填充方式为 SAME '''
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='SAME')

''' 定义第二个偏置项，形状为 [64]，初始值为 0.1 '''
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))

''' 应用 ReLU 激活函数 '''
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))

''' 应用最大池化操作，池化窗口大小为 3x3，步长为 2，填充方式为 SAME '''
poo2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

''' 将池化层输出展平为一维向量 '''
reshape = tf.reshape(poo2, [batch_size, -1])

''' 获取展平后向量的维度 '''
dim = reshape.get_shape()[1].value

''' 定义第一个全连接层的权重，形状为 [dim, 384]，标准差为 0.04，L2 正则化系数为 0.004 '''
weight1 = variable_with_weight_loss([dim, 384], stddev=0.04, w1=0.004)

''' 定义第一个全连接层的偏置项，形状为 [384]，初始值为 0.1 '''
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))

''' 应用 ReLU 激活函数 '''
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

''' 定义第二个全连接层的权重，形状为 [384, 192]，标准差为 0.04，L2 正则化系数为 0.004 '''
weight2 = variable_with_weight_loss([384, 192], stddev=0.04, w1=0.004)

''' 定义第二个全连接层的偏置项，形状为 [192]，初始值为 0.1 '''
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))

''' 应用 ReLU 激活函数 '''
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

''' 定义输出层的权重，形状为 [192, 10]，标准差为 1/192.0，不使用 L2 正则化 '''
weight3 = variable_with_weight_loss([192, 10], stddev=1/192.0, w1=0.0)

''' 定义输出层的偏置项，形状为 [10]，初始值为 0.1 '''
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))

''' 计算最终输出结果 '''
result = tf.add(tf.matmul(local4, weight3), fc_bias3)

''' 计算交叉熵损失 '''
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))

''' 获取所有 L2 正则化损失的总和 '''
weight_with_l2_loss = tf.add_n(tf.get_collection('losses'))

''' 计算总损失，包括交叉熵损失和 L2 正则化损失 '''
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss

''' 定义优化器，使用 Adam 优化器，学习率为 1e-3 '''
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

''' 定义 top-k 准确率计算操作 '''
top_k_op = tf.nn.in_top_k(result, y, 1)

''' 初始化所有变量 '''
init_op = tf.global_variables_initializer()

''' 开始会话 '''
with tf.Session() as sess:
    ''' 运行初始化操作 '''
    sess.run(init_op)

    ''' 启动队列运行器 '''
    tf.train.start_queue_runners()

    ''' 进行训练循环 '''
    for step in range(max_steps):
        ''' 记录开始时间 '''
        start_time = time.time()

        ''' 尝试加载一批训练数据 '''
        try:
            image_batch, label_batch = sess.run([images_train, labels_train])
            ''' 打印当前步骤和加载的数据信息 '''
            print(f"Step {step}: Loaded batch of images and labels.")
            print(f"Image batch shape: {image_batch.shape}, Label batch shape: {label_batch.shape}")
            print(f"First image: {image_batch[0]}")
            print(f"First label: {label_batch[0]}")
        except Exception as e:
            print(f"Error at step {step}: {e}")

        ''' 运行训练操作并获取损失值 '''
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y: label_batch})

        ''' 计算持续时间 '''
        duration = time.time() - start_time

        ''' 每 100 步打印一次训练信息 '''
        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d: loss: %.2f (%.1f examples/sec; %.3f sec/batch" % (step, loss_value, examples_per_sec, sec_per_batch))

        ''' 计算评估批次的数量 '''
        num_batch = int(math.ceil(num_exaples_for_eval / batch_size))
        true_count = 0
        total_sample_count = num_batch * batch_size

        ''' 进行评估循环 '''
        for j in range(num_batch):
            ''' 加载一批测试数据 '''
            image_batch, label_batch = sess.run([images_test, labels_test])

            ''' 运行 top-k 准确率计算操作并获取预测结果 '''
            predictions = sess.run([top_k_op], feed_dict={x: image_batch, y: label_batch})

            ''' 累加正确的预测数量 '''
            true_count += np.sum(predictions)

        ''' 打印评估结果 '''
        print("precision @ 1 = %.3f" % (true_count / total_sample_count))
