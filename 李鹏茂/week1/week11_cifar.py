import tensorflow as tf
import numpy as np
import os
import time
import math
import week11_Cifar10_data as Cifar10_data  # 导入你写的模块

# CIFAR-10类别数量
num_classes = 10
batch_size = 100
num_examples_pre_epoch_for_train = 50000
num_examples_pre_epoch_for_eval = 10000
data_dir = "Cifar_data/cifar-10-batches-bin"


# 构建神经网络模型
def build_model(x):
    # # 第一个卷积层
    # kernel1 = tf.Variable(tf.truncated_normal([5, 5, 3, 64], stddev=5e-2))
    # conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")
    # bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
    # relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
    # pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
    kernel1=tf.Variable(tf.truncated_normal([5,5,3,64],stddev=0.001))
    conv1=tf.nn.conv2d(x,kernel1,[1,1,1,1],padding='SAME')
    bias1=tf.Variable(tf.constant(0.0,shape=[64]))
    relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias1))
    pool1 =tf.nn.max_pool(relu1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
    # 第二个卷积层
    kernel2 = tf.Variable(tf.truncated_normal([5, 5, 64, 64], stddev=5e-2))
    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
    bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
    relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
    pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

    # # 扁平化操作
    # reshape = tf.reshape(pool2, [batch_size, -1])
    # dim = reshape.get_shape()[1].value  # 获取 reshape 后的第二个维度的值
    reshape = tf.reshape(pool2,[batch_size, -1])
    print(reshape.shape)
    dim = reshape.get_shape()[1].value
    print(dim)
    # 全连接层1
    # weight1 = tf.Variable(tf.truncated_normal([dim, 384], stddev=0.04))
    # fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
    # fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
    weight1 = tf.Variable(tf.truncated_normal([dim, 384], stddev=0.04))
    print(weight1.shape)
    fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
    fc_1 =tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)
    # 全连接层2
    weight2 = tf.Variable(tf.truncated_normal([384, 192], stddev=0.04))
    fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
    local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

    # 全连接层3 (输出层)
    weight3 = tf.Variable(tf.truncated_normal([192, num_classes], stddev=1 / 192.0))
    fc_bias3 = tf.Variable(tf.constant(0.1, shape=[num_classes]))
    result = tf.add(tf.matmul(local4, weight3), fc_bias3)
    print(result.shape)
    return result


# 计算损失和准确率
def compute_loss_and_accuracy(logits, y_):
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(y_, tf.int64))
    # loss = tf.reduce_mean(cross_entropy)
    print(y_)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.cast(y_,tf.int64))
    loss = tf.reduce_mean(cross_entropy)
    # 计算准确率
    correct_prediction = tf.equal(tf.argmax(logits, 1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return loss, accuracy


# 训练神经网络
def train():

    image_test,lable_test = Cifar10_data.inputs(data_dir, batch_size, distorted=True)
    image_train, lable_train = Cifar10_data.inputs(data_dir, batch_size, distorted=False)
    x =tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
    y = tf.placeholder(tf.int64, [batch_size])
    logits = build_model(x)
    loss,accuracy = compute_loss_and_accuracy(logits, y)
    OPTIMIZER = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    INIT=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(INIT)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners()
        for i in range(max_steps):
            batch_x, batch_y = sess.run([image_train, lable_train])
            _, loss2 = sess.run([OPTIMIZER, loss], feed_dict={x:batch_x,y:batch_y})
            if i % 100 == 0:
                print(f"Step {i}, Loss: {loss2}")
            if i % 1000 == 0:
                test_images, test_labels = sess.run([image_test,lable_test])
                acc = sess.run(accuracy, feed_dict={x: test_images, y: test_labels})
                print(f"Step {i}, Accuracy: {acc}")
        coord.request_stop()
        coord.join(threads)
if __name__ == "__main__":
    max_steps = 4000  # 最大训练步数
    train()
