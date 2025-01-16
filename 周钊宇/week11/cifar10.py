import tensorflow.compat.v1 as tf
import cifar_data
import time
import math
import numpy as np


tf.disable_v2_behavior()
batch_size = 100
num_examples_for_eval = 10000
max_steps = 4000
data_dir = '/Users/zhouzhaoyu/Desktop/ai/week11/cifar_data/cifar-10-batches-bin'

def variable_with_weight_loss(shape, mean, stddev, w1):
    """用w1控制l2 loss
    输入参数：
    1. shape:权重的shape
    2. mean: 均值
    3. stddev: 标准差
    4. w1: 用来控制l2loss

    返回值：输出一个权重矩阵
    """
    var = tf.truncated_normal(shape, mean, stddev)
    if w1 is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), w1, name = "weights_loss")
        tf.add_to_collection("losses", weight_loss)
    return var

image_train, label_train = cifar_data.data_inputs(data_dir, batch_size, True)
image_test, label_test = cifar_data.data_inputs(data_dir, batch_size, None)

x = tf.placeholder(tf.float32, [batch_size, 24, 24 ,3])
y = tf.placeholder(tf.int32, [batch_size])

#创建conv1卷积层,卷积核是5*5，64个卷积核
kernel1 = variable_with_weight_loss([5,5,3,64], 0, 5e-2, w1 = 0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1,1,1,1], padding='SAME')
bias1 = tf.Variable(tf.constant(0.0, shape = [64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

#创建卷积层conv2， 卷积核5*5，64个卷积核
kernel2 = variable_with_weight_loss([5,5,64,64], 0, 5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding='SAME')
bias2 = tf.Variable(tf.constant(0.0, shape = [64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

#当前输出张量为
reshape = tf.reshape(pool2, [batch_size, -1]) # [100, 2304]
dim = reshape.get_shape()[1].value   #2304
# print(dim)


#建立全连接层FC1
weight1 = variable_with_weight_loss(shape=[dim, 384], mean=0, stddev=0.04, w1=0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshape, weight1), fc_bias1))   # [100,384]

#建立全连接层FC2
weight2 = variable_with_weight_loss(shape=[384,192], mean=0, stddev=0.04, w1=0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1, weight2), fc_bias2))  #[100,192]

#建立全连接层FC3
weight3 = variable_with_weight_loss(shape=[192, 10], mean=0, stddev=0.04, w1=0.004)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
fc3 = tf.nn.bias_add(tf.matmul(fc2, weight3), fc_bias3) #[100,10]

#计算损失softmax + crossentropy tf.nn.sparse_softmax_cross_entropy_with_logits()用于生成稀疏的分类损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y, tf.int64), logits=fc3)

weights_with_l2loss = tf.add_n(tf.get_collection("losses"))
final_loss = tf.reduce_mean(cross_entropy) + weights_with_l2loss

#反向传播，优化
train_op = tf.train.AdamOptimizer(0.001).minimize(final_loss)

topk_op = tf.nn.in_top_k(fc3, y,1) #返回一个布尔值用来判断结果中的最大值是否在标签y中

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # tf.train.start_queue_runners() ## 启动线程操作

    for step in range(max_steps):
        start_time = time.time()
        image_batch, label_batch = sess.run([image_train, label_train])
        _,loss_value = sess.run([train_op, final_loss], feed_dict={x:image_batch, y:label_batch})
        duration = time.time() - start_time


        if step % 100 ==0:
            examples_per_sec = batch_size/duration
            sec_per_batch = float(duration)
            print("step%d, loss=%.2f(%.1f examples/sec, %.3f sec/batch)")%(step, loss_value, examples_per_sec, sec_per_batch)
    
    num_batch = int(math.ceil(num_examples_for_eval/batch_size))
    true_count = 0
    total_count = num_batch * batch_size

    for j in range(num_batch):
        image_batch, label_batch = sess.run([image_test, label_test])
        predicts = sess.run([topk_op], feed_dict = {x:image_batch, y:label_batch})
        true_count += np.sum(predicts)

    print("accuracy = %.3f%%"%((true_count/total_count)*100))