#-*- coding:utf-8 -*-
# author: 王博然
import tensorflow as tf
import numpy as np
import time
import math
import cifar10_data

data_dir = "Cifar_data/cifar-10-batches-bin"
max_steps = 400#0
batch_size = 100
num_examples_for_eval = 10000 # 评估

# 1.使用参数w1控制L2 loss大小
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection("losses", weights_loss)
    return var

# 读取数据, 训练数据会做增强, 测试数据则不会
images_train, labels_train = cifar10_data.inputs(data_dir, batch_size, True)
images_test, labels_test = cifar10_data.inputs(data_dir, batch_size, None)
# 创建两个placeholder，用于在训练或评估时提供输入的数据和对应的标签值
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y_target = tf.placeholder(tf.int32, [batch_size])

# 创建第一个卷积层
kernel1 = variable_with_weight_loss(shape=[5,5,3,64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1,1,1,1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5,5,64,64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

# 全连接的过渡, 需要拍扁
reshape = tf.reshape(pool2, [batch_size, -1])  # 自动计算, 将pool2的三维结构拉直为一维结构
dim = reshape.get_shape()[1].value             # reshape之后的第二个维度

# 建立第一个全连接
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 建立第二个全连接
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1/192.0, w1=0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

# 计算损失, 包含权重参数的正则化损失 和 交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_target, tf.int64))
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_k_op = tf.nn.in_top_k(result, y_target, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    # 启动线程操作
    tf.train.start_queue_runners()

    # 每隔 100s step会计算并展示当前的loss、每秒钟能训练的样本个数、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x:image_batch, y_target:label_batch})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size/duration
            print("step %d, loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % \
                  (step, loss_value, examples_per_sec, float(duration)))

    # 计算最终的正确率
    num_batch = math.ceil(num_examples_for_eval/batch_size)
    true_count = 0

    for i in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x:image_batch, y_target:label_batch})
        true_count += np.sum(predictions)

    print("accuracy = %.3f%%" % ((true_count/num_examples_for_eval) * 100))
