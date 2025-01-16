import math

import numpy as np
import tensorflow as tf
from tensorflow import shape
from tensorflow.contrib.labeled_tensor import reshape
from tensorflow.python.ops.distributions.kullback_leibler import cross_entropy

import cifar10_data

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "cifar_data/cifar-10-batches-bin"

def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")
        tf.add_to_collection("losses", weights_loss)
    return var

images_train, labels_train = cifar10_dataa.inputs(data_dir, batch_size, True)
images_test, labels_test = cifar10_dataa.inputs(data_dir, batch_size, True)

# 用于输入数据和标签
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.int32, [batch_size])

# 卷积层1
kernel1 = variable_with_weight_loss([5, 5, 3, 64], stddev=5e-2, w1=0.0)
conv1 = tf.nn.conv2d(x, kernel1, strides=[1, 1, 1, 1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
print("pool1.shape", pool1.shape)

# 卷积层2
kernel2 = variable_with_weight_loss([5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.0, shape=[64]))
relu2 = tf.nn.relu(tf.add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
print("pool2.shape", pool2.shape)

# 全连接层
reshape = tf.reshape(pool2, [batch_size, -1])
print("reshape.shape", reshape.shape)
dim = reshape.get_shape()[1].value

# 全连接层1
weight1 = variable_with_weight_loss([dim, 384], stddev=4e-2, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 全连接层2
weight2 = variable_with_weight_loss([384, 192], stddev=4e-2, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc2 = tf.nn.relu(tf.matmul(fc1, weight2) + fc_bias2)

# 全连接层3
weight3 = variable_with_weight_loss([192, 10], stddev=4e-2, w1=0.004)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(fc2, weight3), fc_bias3)

# 计算cross_entropy
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y, tf.int64))

weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

top_k_op = tf.nn.in_top_k(result, y, 1)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners()

    for step in range(max_steps):
        image_batch, label_batch = sess.run([images_train, labels_train])
        _, loss_value = sess.run([train_op, loss], feed_dict={x:image_batch, y:label_batch})

        if step % 100 == 0:
            print("step %d, loss %.2f"%(step, loss_value))

    num_batch = int(math.ceil(num_examples_for_eval / batch_size))
    right_count = 0
    total_sample_count = num_batch * batch_size

    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run(top_k_op, feed_dict={x:image_batch, y:label_batch})
        right_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((right_count / total_sample_count) * 100))
