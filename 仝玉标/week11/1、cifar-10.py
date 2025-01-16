import tensorflow as tf
import numpy as np
import time
import math
import cifar10_data_sakura

max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000  # 用来评估的样本数
data_dir = "Cifar_data/cifar-10-batches-bin"  # 数据集目录


def variable_with_weight_loss(shape, stddev, w1):  # 函数是用于创建带有L2正则化损失的变量
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))  # 使用截断正态分布初始化一个变量,标准差为stddev
    if w1 is not None:  # 如果w1不为None，则计算L2损失
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name="weights_loss")  # 计算变量的L2损失，并乘以w1（w1是正则化系数）得到最终的L2正则化损失
        tf.add_to_collection("losses", weights_loss)  # 将L2正则化损失添加到名为"losses"的集合中，从而能够在训练过程中被自动计算并添加到总损失中
    return var


# 训练集要进行图像增强预处理，而测试集不用
images_train, labels_train = cifar10_data_sakura.inputs(data_dir=data_dir, batch_size=batch_size, distorted=True)
images_test, labels_test = cifar10_data_sakura.inputs(data_dir=data_dir, batch_size=batch_size, distorted=None)

x = tf.placeholder(tf.float32, [batch_size, 24, 24, 3])  # 动态存储输入数据
y_ = tf.placeholder(tf.int32, [batch_size])  # 动态存储数据标签

# 第一个卷积层，卷积核大小为5*5、通道数=3、卷积核个数=64
kernel1 = variable_with_weight_loss(shape=[5, 5, 3, 64], stddev=5e-2, w1=0.0)  # 正则化系数w1为0，不使用L2正则化
conv1 = tf.nn.conv2d(x, kernel1, [1, 1, 1, 1], padding="SAME")  # [1, 1, 1, 1]的顺序依次为[batch_size, H, W, C]
bias1 = tf.Variable(tf.constant(0.0, shape=[64]))  # bias1的个数 = conv1的输出节点个数
relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")  # pooling大小为3*3、步长是2*2

# 创建第二个卷积层
kernel2 = variable_with_weight_loss(shape=[5, 5, 64, 64], stddev=5e-2, w1=0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")

# 卷积层结果输入到FC要，要先reshape变成一维的
reshape = tf.reshape(pool2, [batch_size, -1])  # 这里面的-1代表将pool2的三维结构拉直为一维结构
dim = reshape.get_shape()[1].value  # get_shape()获取的是一维扁平化后的长度大小

# 建立第一个全连接层
weight1 = variable_with_weight_loss(shape=[dim, 384], stddev=0.04, w1=0.004)  # 使用L2正则化
fc_bias1 = tf.Variable(tf.constant(0.1, shape=[384]))
fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

# 建立第二个全连接层
weight2 = variable_with_weight_loss(shape=[384, 192], stddev=0.04, w1=0.004)  # 使用L2正则化
fc_bias2 = tf.Variable(tf.constant(0.1, shape=[192]))
local4 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

# 建立第三个全连接层
weight3 = variable_with_weight_loss(shape=[192, 10], stddev=1 / 192.0, w1=0.0)  # 不使用L2正则化
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[10]))
result = tf.add(tf.matmul(local4, weight3), fc_bias3)

# 计算损失，包括权重参数的正则化损失和交叉熵损失
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result, labels=tf.cast(y_, tf.int64))
'''
交叉熵损失的工作流程：
1.对于每个样本，logits 张量中的对应行会通过一个 softmax 函数进行归一化，得到一个概率分布。
2.根据 labels 张量中提供的真实类别索引，从概率分布中选取对应类别的概率值。
3.计算负对数似然，即 -log(selected_probability)，作为该样本的损失值。
！！！重要事项：tf.nn.sparse_softmax_cross_entropy_with_logits 函数内部已经对 logits 进行了 softmax 计算
'''
weights_with_l2_loss = tf.add_n(tf.get_collection("losses"))    # 计算并汇总所有被添加到 "losses" 集合中的损失值
loss = tf.reduce_mean(cross_entropy) + weights_with_l2_loss

# 使用 Adam 优化器并最小化损失函数
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)  # tf.train.AdamOptimizer操作执行后会更新模型的参数以最小化损失，“1e-3”是学习率

# 函数tf.nn.in_top_k()用来计算输出结果中top k的准确率，函数默认的k值是1，即top 1的准确率，也就是输出分类准确率最高时的数值
top_k_op = tf.nn.in_top_k(result, y_, 1)

init_op = tf.global_variables_initializer()     # 执行后，tf.global_variables_initializer()会遍历图中所有的全局变量，并将它们初始化为它们的默认值

with tf.Session() as sess:
    sess.run(init_op)

    # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
    tf.train.start_queue_runners()

    # 每隔100step会计算并展示当前的loss、每秒钟能训练的样本数量、以及训练一个batch数据所花费的时间
    for step in range(max_steps):
        start_time = time.time()    # 记录当前时间，用于计算每个训练步骤的持续时间
        image_batch, label_batch = sess.run([images_train, labels_train])   # 获取下一个图像批次和标签批次
        _, loss_value = sess.run([train_op, loss], feed_dict={x: image_batch, y_: label_batch})     #运行训练操作和损失计算操作，并传入当前批次的图像和标签作为输入
        duration = time.time() - start_time     # 计算当前训练步骤的持续时间

        if step % 100 == 0:     # 每隔100step会记录一次
            examples_per_sec = batch_size / duration
            sec_per_batch = float(duration)
            print("step %d,loss=%.2f(%.1f examples/sec;%.3f sec/batch)" % (
                step, loss_value, examples_per_sec, sec_per_batch))

    # 计算最终的正确率
    num_batch = int(math.ceil(num_examples_for_eval / batch_size))  # math.ceil()函数用于求整
    true_count = 0
    total_sample_count = num_batch * batch_size

    # 在一个for循环里面统计所有预测正确的样例个数
    for j in range(num_batch):
        image_batch, label_batch = sess.run([images_test, labels_test])
        predictions = sess.run([top_k_op], feed_dict={x: image_batch, y_: label_batch})
        true_count += np.sum(predictions)

    # 打印正确率信息
    print("accuracy = %.3f%%" % ((true_count / total_sample_count) * 100))
