import tensorflow as tf
import numpy as np
import time
import math
import Cifar10_data


# 基础配置
class Config:
    def __init__(self):
        self.max_steps = 4000  # 训练总步数
        self.batch_size = 100  # 每批数据量
        self.eval_samples = 10000  # 测试样本数量
        self.data_dir = "Cifar_data/cifar-10-batches-bin"  # 数据目录


# 创建网络权重，并添加L2正则化
def create_weight(shape, std_dev, weight_loss_factor):
    """
    创建一个带有权重损失的变量
    shape: 权重形状
    std_dev: 标准差
    weight_loss_factor: 权重损失系数
    """
    weight = tf.Variable(tf.truncated_normal(shape, stddev=std_dev))

    # 如果需要添加L2正则化
    if weight_loss_factor is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(weight), weight_loss_factor)
        tf.add_to_collection("losses", weight_loss)

    return weight


# 创建卷积层
def create_conv_layer(input_data, kernel_shape, bias_value):
    """
    创建一个卷积层
    input_data: 输入数据
    kernel_shape: 卷积核形状
    bias_value: 偏置值
    """
    kernel = create_weight(kernel_shape, std_dev=5e-2, weight_loss_factor=0.0)
    conv = tf.nn.conv2d(input_data, kernel, [1, 1, 1, 1], padding="SAME")
    bias = tf.Variable(tf.constant(bias_value, shape=[kernel_shape[-1]]))
    relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
    return relu


# 创建池化层
def create_pool_layer(input_data):
    """
    创建一个最大池化层
    input_data: 输入数据
    """
    return tf.nn.max_pool(
        input_data, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME"
    )


# 创建全连接层
def create_fc_layer(input_data, output_size, std_dev, weight_loss_factor):
    """
    创建一个全连接层
    input_data: 输入数据
    output_size: 输出维度
    std_dev: 标准差
    weight_loss_factor: 权重损失系数
    """
    input_size = input_data.get_shape()[1].value
    weight = create_weight([input_size, output_size], std_dev, weight_loss_factor)
    bias = tf.Variable(tf.constant(0.1, shape=[output_size]))
    return tf.nn.relu(tf.matmul(input_data, weight) + bias)


# 构建完整的网络模型
def build_model(config):
    """
    构建完整的CNN模型
    config: 配置参数
    """
    # 创建输入占位符
    x = tf.placeholder(tf.float32, [config.batch_size, 24, 24, 3])
    y = tf.placeholder(tf.int32, [config.batch_size])

    # 第一个卷积和池化层
    conv1 = create_conv_layer(x, [5, 5, 3, 64], bias_value=0.0)
    pool1 = create_pool_layer(conv1)

    # 第二个卷积和池化层
    conv2 = create_conv_layer(pool1, [5, 5, 64, 64], bias_value=0.1)
    pool2 = create_pool_layer(conv2)

    # 展平数据
    flatten = tf.reshape(pool2, [config.batch_size, -1])

    # 全连接层
    fc1 = create_fc_layer(flatten, 384, 0.04, 0.004)
    fc2 = create_fc_layer(fc1, 192, 0.04, 0.004)

    # 输出层
    final_weight = create_weight([192, 10], 1 / 192.0, 0.0)
    final_bias = tf.Variable(tf.constant(0.1, shape=[10]))
    logits = tf.add(tf.matmul(fc2, final_weight), final_bias)

    return x, y, logits


# 计算损失
def calculate_loss(logits, labels):
    """
    计算模型的总损失
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, tf.int64)
    )
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    regular_losses = tf.add_n(tf.get_collection("losses"))
    total_loss = cross_entropy_mean + regular_losses
    return total_loss


# 主训练函数
def train_model():
    # 初始化配置
    config = Config()

    # 准备数据
    images_train, labels_train = Cifar10_data.inputs(
        data_dir=config.data_dir, batch_size=config.batch_size, distorted=True
    )

    images_test, labels_test = Cifar10_data.inputs(
        data_dir=config.data_dir, batch_size=config.batch_size, distorted=None
    )

    # 构建模型
    x, y, logits = build_model(config)

    # 计算损失
    loss = calculate_loss(logits, y)

    # 创建训练操作
    train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # 计算准确率
    accuracy = tf.nn.in_top_k(logits, y, 1)

    # 开始训练
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.train.start_queue_runners()

        # 训练循环
        for step in range(config.max_steps):
            start_time = time.time()

            # 获取一批训练数据
            image_batch, label_batch = sess.run([images_train, labels_train])

            # 训练一步
            _, loss_value = sess.run(
                [train_op, loss], feed_dict={x: image_batch, y: label_batch}
            )

            # 打印训练信息
            if step % 100 == 0:
                duration = time.time() - start_time
                examples_per_sec = config.batch_size / duration
                print(
                    f"步骤 {step}, 损失值 = {loss_value:.2f} "
                    f"({examples_per_sec:.1f} 样本/秒)"
                )

        # 评估模型
        print("开始评估模型...")
        num_batches = math.ceil(config.eval_samples / config.batch_size)
        total_correct = 0
        total_samples = num_batches * config.batch_size

        for _ in range(num_batches):
            image_batch, label_batch = sess.run([images_test, labels_test])
            predictions = sess.run(accuracy, feed_dict={x: image_batch, y: label_batch})
            total_correct += np.sum(predictions)

        final_accuracy = (total_correct / total_samples) * 100
        print(f"最终准确率 = {final_accuracy:.2f}%")


if __name__ == "__main__":
    train_model()
