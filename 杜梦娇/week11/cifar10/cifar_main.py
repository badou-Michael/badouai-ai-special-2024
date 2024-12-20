import time
import math
import numpy as np
import cifar_input
import tensorflow as tf


# 定义变量
max_step = 2000
batch_size = 120
eval_date_num = 10000
data_dir = "cifar-10-batches-bin"

# 导入数据
train_images, train_labels = cifar_input.input_data(data_dir, batch_size, True)
test_images, test_labels = cifar_input.input_data(data_dir, batch_size, None)
# 定义输入和输出的placeholder
X_placeholder = tf.placeholder(tf.float32, [batch_size, 28, 28, 3])
Y_placeholder = tf.placeholder(tf.int32, [batch_size])

#创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape,stddev,w1):
    var=tf.Variable(tf.truncated_normal(shape,stddev=stddev))
    if w1 is not None:
        weights_loss=tf.multiply(tf.nn.l2_loss(var),w1,name="weights_loss")
        tf.add_to_collection("losses",weights_loss)
    return var

# 定义模型结构(第一个卷积块)
kernel1 = variable_with_weight_loss([5,5,3,64], stddev=5e-2, w1=0.002)
conv1 = tf.nn.conv2d(X_placeholder, kernel1, [1,1,1,1], padding="SAME")
B1 = tf.Variable(tf.constant(0.01, shape=[64]))
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, B1))
pool1 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

# 定义模型结构(第二个卷积块)
kernel2 = variable_with_weight_loss([3,3,64,64], stddev=5e-2, w1=0.002)
conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding="SAME")
B2 = tf.Variable(tf.constant(0.01, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, B2))
pool2 = tf.nn.max_pool(relu1, ksize=[1,3,3,1], strides=[1,2,2,1], padding="SAME")

# 数据拍扁
data_reshaped = tf.reshape(pool2, [batch_size,-1])
shape1 = data_reshaped.get_shape()[1].value

# 定义模型结构(第一个FC)
weight1 = variable_with_weight_loss([shape1, 512], stddev=5e-2, w1=0.002)
bias1 = tf.Variable(tf.constant(0.01, shape=[512]))
fc1 = tf.nn.relu(tf.matmul(data_reshaped, weight1) + bias1)

# 定义模型结构(第二个FC)
weight2 = variable_with_weight_loss([512,128], stddev=5e-2, w1=0.002)
bias2 = tf.Variable(tf.constant(0.01, shape=[128]))
fc2 = tf.nn.relu(tf.matmul(fc1, weight2) + bias2)

# 定义模型结构(第二个FC)
weight3 = variable_with_weight_loss([128,10], stddev=5e-2, w1=0.002)
bias3 = tf.Variable(tf.constant(0.01, shape=[10]))
fc3 = tf.matmul(fc2, weight3) + bias3

# 定义损失函数
cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=fc3,labels=tf.cast(Y_placeholder, tf.int64))
# 加上权重的损失函数
weights_with_l2_loss=tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy_loss)+weights_with_l2_loss
# 定义optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=0.002).minimize(loss)
# 计算输出结果中top k的准确率
top_k_op = tf.nn.in_top_k(fc3, Y_placeholder, 1)

# 启动会话
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    # 启动线程操作
    tf.train.start_queue_runners()

    # 训练模型
    for step in range(max_step):
        start_time = time.time()
        images_batch, label_batch = sess.run([train_images, train_labels])
        _, loss_value = sess.run([optimizer, loss], feed_dict={X_placeholder: images_batch, Y_placeholder: label_batch})
        time_run = time.time() - start_time

        # 每隔一百个epoch打印结果
        if (step + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{max_step}], Loss: {loss_value:.2f}, ({examples_per_sec:.1f} examples/sec; {sec_per_batch:.3f} sec/batch)')
    # 计算准确率
    num_batch = int(math.ceil(eval_date_num/batch_size))
    num_ture = 0
    num_total_sample = num_batch * batch_size

    for i in range(num_batch):
        images_batch, label_batch = sess.run([test_images, test_labels])
        prediction_value = sess.run([top_k_op], feed_dict={X_placeholder: images_batch, Y_placeholder: label_batch})
        num_ture += np.sum(prediction_value)

    print("ACC: %.4f%%" % ((num_ture/num_total_sample) * 100))
