import tensorflow as tf

# 参数设置
input_nodes = 784  # 输入节点数（28x28）
hidden_nodes = 200  # 隐藏层节点数
output_nodes = 10  # 输出节点数（0-9分类）
learning_rate = 0.1

# 构建计算图
x = tf.placeholder(tf.float32, [None, input_nodes])  # 输入占位符
y = tf.placeholder(tf.float32, [None, output_nodes])  # 输出占位符

# 定义权重和偏置
W1 = tf.Variable(tf.random_normal([input_nodes, hidden_nodes], stddev=0.1))
b1 = tf.Variable(tf.zeros([hidden_nodes]))
W2 = tf.Variable(tf.random_normal([hidden_nodes, output_nodes], stddev=0.1))
b2 = tf.Variable(tf.zeros([output_nodes]))

# 前向传播
hidden_output = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
final_output = tf.nn.softmax(tf.matmul(hidden_output, W2) + b2)

# 损失函数和优化器
loss = tf.reduce_mean(-tf.reduce_sum(y * tf.log(final_output), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

# 模拟训练数据
train_X = tf.random.normal([100, 784]).eval(session=tf.Session())  # 100条训练样本
train_Y = tf.one_hot(tf.random.uniform([100], 0, 10, dtype=tf.int32), depth=10).eval(session=tf.Session())

# 模拟测试数据
test_X = tf.random.normal([10, 784]).eval(session=tf.Session())  # 10条测试样本

# 训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # 初始化变量
    for epoch in range(10):  # 训练10个epoch
        sess.run(optimizer, feed_dict={x: train_X, y: train_Y})

    # 推理
    predictions = sess.run(final_output, feed_dict={x: test_X})
    print("预测结果（概率分布）：", predictions)
    print("预测类别：", tf.argmax(predictions, axis=1).eval())
