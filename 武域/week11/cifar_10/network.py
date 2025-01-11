import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data

batch_size = 100
eval_size = 10000
train_size = 50000
data_dir = 'cifar_data/cifar-10-batches-bin'
epochs = int(input("How many epochs? "))
max_steps = int(epochs * train_size / batch_size)


def var_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weights_loss')
        tf.add_to_collection('loss', weights_loss)
    return var


# Load the data, where the train data is distorted and the test data is not
train_images, train_labels = Cifar10_data.input_fn(data_dir=data_dir, batch_size=batch_size, distorted=True)
test_images, test_labels = Cifar10_data.input_fn(data_dir=data_dir, batch_size=batch_size, distorted=False)

# Define place holders
x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])
y = tf.placeholder(tf.uint8, [batch_size])

# create first convolution layer, where shape is defined as [kh, kw, ci, co]
k1 = var_with_weight_loss([5, 5, 3, 64], 0.01, 0.00)
conv1 = tf.nn.conv2d(x, k1, strides=[1, 1, 1, 1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# crete second convolution layer
k2 = var_with_weight_loss([5, 5, 64, 64], 0.01, 0.00)
conv2 = tf.nn.conv2d(pool1, k2, strides=[1, 1, 1, 1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

reshape = tf.reshape(pool2, [batch_size, -1])
dim = reshape.get_shape()[1].value

# create first full connection layer
w1 = var_with_weight_loss([dim, 384], 0.04, 0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, w1) + fc_bias1)

# create 2nd full connection layer
w2 = var_with_weight_loss([384, 192], 0.04, 0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc_2 = tf.nn.relu(tf.matmul(fc_1, w2) + fc_bias2)

# create 3rd full connection layer
w3 = var_with_weight_loss([192, 10], 1 / 192, 0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
res = tf.add(tf.matmul(fc_2, w3), fc_bias3)

# calculate loss
cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=res,labels=tf.cast(y,tf.int64))
weight_with_l2_loss = tf.add_n(tf.get_collection('loss'))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss

train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)
top_1 = tf.nn.in_top_k(res, tf.cast(y, tf.int32), k=1)
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    tf.train.start_queue_runners(sess=sess)
    for step in range(max_steps):
        start_time = time.time()
        images, labels = sess.run([train_images, train_labels])
        _, loss_value = sess.run([train_op, loss], feed_dict={x: images, y: labels})
        duration = time.time() - start_time

        if step % 100 == 0:
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)"%(step,loss_value,examples_per_sec,sec_per_batch))

    num_batch = int(math.ceil(eval_size / batch_size))
    true_num = 0
    total_num = batch_size * num_batch

    for i in range(num_batch):
        image, label = sess.run([test_images, test_labels])
        prediction = sess.run(top_1, feed_dict={x: image, y: label})
        true_num += np.sum(prediction)

    print("accuracy = %.3f%%" % ((true_num / total_num) * 100))
