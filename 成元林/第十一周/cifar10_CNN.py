import math
import time

import numpy
import tensorflow as tf
import tensorflow.nn as nn
from 成元林.第十一周 import cifar10Data

# 一个批次100张图片
batchSize = 100
# 文件路劲
dir_Path="cifar_data/cifar-10-batches-bin"
testNum = 10000


# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，

def variable_with_weight_loss(shape, stddev, w1):
    # 定义指定形状的随机卷积核
    kernnel = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weight_loss = tf.multiply(nn.l2_loss(kernnel), w1, name="weightLoss")
        tf.add_to_collection("losses", weight_loss)
    return kernnel


trainBatchData, trainBatchLabel = cifar10Data.handleInput(dirPath=dir_Path, batchSize=batchSize, isImageEhanmance=True)
testBatchData, testBatchLabel = cifar10Data.handleInput(dirPath=dir_Path, batchSize=batchSize, isImageEhanmance=False)
x = tf.placeholder(tf.float32, [batchSize, 24, 24, 3])
y = tf.placeholder(tf.int32, [batchSize])

# 第一层卷积
# 获得卷积核
kernel1 = variable_with_weight_loss([5, 5, 3, 64], stddev=0.01, w1=0.0)
# 进行卷积操作
cov1 = nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
# 偏置
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
# 卷积结果加偏置，使用relu增加非线性
cov_result = nn.relu(cov1 + bias1)
# 最大池化
pool1 = nn.max_pool(cov_result, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="max_pool1")

# 第二次卷积
kernel2 = variable_with_weight_loss([5, 5, 64, 64], stddev=0.05, w1=0.0)
cov2 = nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
cov_result2 = nn.relu(tf.add(cov2, bias2))
pool2 = nn.max_pool(cov_result2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME", name="max_pool1")

# 对卷积结果扁平化
cov_final_result = tf.reshape(pool2, shape=[batchSize, -1])  # (100,n)
input_size = cov_final_result.get_shape()[1].value

# 第一层全连接，输入层到隐藏层 (n,384)
w1 = variable_with_weight_loss(shape=[input_size, 384], stddev=0.001, w1=0.0)
# y = wx+b
biasw1 = tf.Variable(tf.constant(0.2, shape=[384]))
result_w1 = tf.matmul(cov_final_result, w1) + biasw1  # (100,384)
relu1 = nn.relu(result_w1, name="relu1")

# 第二层全连接
w2 = variable_with_weight_loss(shape=[384, 100], stddev=0.002, w1=0.0)
biasw2 = tf.Variable(tf.constant(0.1, shape=[100]))
result_w2 = tf.add(tf.matmul(relu1, w2), biasw2)  # (100,100)
relu2 = nn.relu(result_w2, name="relu2")

# 第三层全连接
w3 = variable_with_weight_loss(shape=[100, 10], stddev=0.005, w1=0.0)
biasw3 = tf.Variable(tf.constant(0.1, shape=[10]))
result_w3 = tf.add(tf.matmul(relu2, w3), biasw3)  # (100,10)

# 最后一层全连接激活函数用softmax,
cross_entropy = nn.sparse_softmax_cross_entropy_with_logits(logits=result_w3, labels=tf.cast(y,tf.int64), name="loss")

cross_loss = tf.reduce_mean(cross_entropy)
weightloss = tf.add_n(tf.get_collection("losses"))
total_loss = cross_loss + weightloss

# 优化项
train_op = tf.train.AdamOptimizer(1e-3).minimize(total_loss)
# result_w3获取前k的最大值，并取得最大值相对应的索引，是否包含y中的索引，如果包含，则返回[True]，否则返回[false]
top_1 = nn.in_top_k(result_w3, y,1)

# 初始化
init_op = tf.global_variables_initializer()
epoch = 4000
with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()
    # 开始训练
    for i in range(epoch):
        startime = time.time()
        image_batch, label_batch = sess.run([trainBatchData, trainBatchLabel])
        _, trainloss = sess.run([train_op, total_loss], feed_dict={x: image_batch, y: label_batch})
        endtime = time.time()
        usetime = endtime - startime
        # 每batchSize统计loss,和每秒训练的数量
        if i % batchSize == 0:
            persec_train_num = batchSize / usetime
            bacch_num_sec = float(usetime)
            print("epoch:%d,loss=%.2f,平均每秒训练数量：%.1f,每一代用时：%.2f" % (i, trainloss, persec_train_num, bacch_num_sec))

    # 测试，计算正确率
    test_bacth_num = int(math.ceil(testNum / batchSize))
    trueNum = 0
    testTotal = test_bacth_num * batchSize
    for i in range(test_bacth_num):
        test_image_batch, test_label_batch = sess.run([testBatchData, testBatchLabel])
        predict = sess.run([top_1], feed_dict={x: test_image_batch, y: test_label_batch})
        trueNum += numpy.sum(predict)
    print("accrency:%.2f" % (trueNum / testTotal))
