import tensorflow.compat.v1 as tf
import numpy as np
import time
import os
import lesson11_1_Preprocess_data as Pdata
tf.disable_v2_behavior()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Performance:

    def __init__(self, epochs, batch_size):
        self.epochs = epochs
        self.batch_size = batch_size

        # 读取进来训练数据
        train_data = Pdata.Preprocessing(batch_size, 'train', True)
        self.train_labels, self.train_images = train_data.labels, train_data.images
        # 读取进来测试数据
        test_data = Pdata.Preprocessing(batch_size, 'test', False)
        self.test_labels, self.test_images = test_data.labels, test_data.images

        # 定义两个空tf变量，一是用于在tf计算流中应用tf变量，二是用于向Session传参数
        self.images = tf.placeholder(tf.float32, shape=[100, 24, 24, 3])
        self.labels = tf.placeholder(tf.int32, shape=[100])

        # 编译
        self.conv1 = self.conv1()
        self.fc_train, self.feature_length = self.conv2()
        self.fc1 = self.fc1()
        self.fc2 = self.fc2()
        self.fc3 = self.fc3()
        self.loss = self.calculate_loss()
        self.train_op = self.back_propagation()


    def conv1(self):
        '''
        当前卷积层设置
        卷积、偏置、relu、最大池化
        '''
        # 初始化权重，调用截断高斯分布方法，实现更紧凑的高斯分布权重系数，stddev适用于He初始化方法
        w = tf.Variable(tf.random.truncated_normal([5, 5, 3, 64], stddev=tf.sqrt(2 / self.batch_size)))
        conv = tf.nn.conv2d(self.images, w, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.constant(0.0, shape=[64])
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        return max_pool

    def conv2(self):
        '''
        当前卷积层设置
        卷积、偏置、relu、最大池化
        '''
        # 初始化权重，调用截断高斯分布方法，实现更紧凑的高斯分布权重系数，stddev适用于He初始化方法
        w = tf.Variable(tf.random.truncated_normal([5, 5, 64, 64], stddev=tf.sqrt(2 / self.batch_size)))
        conv = tf.nn.conv2d(self.conv1, w, strides=[1, 1, 1, 1], padding='SAME')
        bias = tf.constant(0.1, shape=[64])
        relu = tf.nn.relu(tf.nn.bias_add(conv, bias))
        max_pool = tf.nn.max_pool(relu, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

        # 对结果进行一定的处理，后续进入fc网络
        fc_train = tf.reshape(max_pool, [self.batch_size, -1])
        feature_length = fc_train.get_shape().as_list()[1]

        return fc_train, feature_length

    def fc1(self):
        w = tf.Variable(tf.random.truncated_normal([self.feature_length, 512], stddev=tf.sqrt(2 / self.feature_length)))
        bias = tf.constant(0.1, shape=[512])
        relu = tf.nn.relu(tf.matmul(self.fc_train, w) + bias)

        return relu, w

    def fc2(self):
        w = tf.Variable(tf.random.truncated_normal([512, 128], stddev=tf.sqrt(2 / 512)))
        bias = tf.constant(0.1, shape=[128])
        relu = tf.nn.relu(tf.matmul(self.fc1[0], w) + bias)

        return relu, w

    def fc3(self):
        # 第三层fc就不用激活函数了，因为后续的计算损失函数是会利用softmax进行计算
        w = tf.Variable(tf.random.truncated_normal([128, 10], stddev=tf.sqrt(2 / 128)))
        bias = tf.constant(0.1, shape=[10])
        output = tf.matmul(self.fc2[0], w) + bias

        return output

    def calculate_loss(self):
        # 稀疏分类交叉熵，包含了分类用的softmax函数
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=self.labels,
            logits=self.fc3,
        )

        # 把fc的第一层第二层的权重计算l2损失
        fc1_l2 = tf.nn.l2_loss(self.fc1[1])
        fc2_l2 = tf.nn.l2_loss(self.fc2[1])
        # 把l2损失存入tf的特定收集器中
        tf.add_to_collection("losses", fc1_l2)
        tf.add_to_collection("losses", fc2_l2)
        # 理论上是要让每个l2损失与主损失相加，但加法交换，先求和收集器中的，再与主损失相加
        sum_l2 = tf.add_n(tf.get_collection("losses"))
        # 求最终的损失，交叉熵因为是一个矩阵，需要算一个主平均值，然后才能与l2损失相加求得一个主损失
        loss = tf.reduce_sum(tf.reduce_mean(cross_entropy) + 0.005 * sum_l2)

        return loss

    def back_propagation(self):
        # 设置使用adam的优化器，并实现权重更新
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001,
        )
        train_op = optimizer.minimize(self.loss)

        return train_op

    def process(self):
        true_count = 0
        init_op = tf.global_variables_initializer()
        # 展示正确率最高的分类
        top = tf.nn.in_top_k(self.fc3, self.labels, 1)

        with tf.Session() as sess:
            # 初始化变量
            sess.run(init_op)
            # 启动线程操作，这是因为之前数据增强的时候使用train.shuffle_batch()函数的时候通过参数num_threads()配置了16个线程用于组织batch的操作
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(self.epochs):
                start_time = time.time()
                # 将之前记录训练数据和答案的两个操作对象，通过sess.run启动，并将值赋给两个新的变量
                train_images, train_labels = sess.run([self.train_images, self.train_labels])
                # 有了训练集的数据，就可以运行最后的train_op了，此时我还想把loss取出来，所以也把loss写上
                _, loss_value = sess.run([self.train_op, self.loss], feed_dict={self.images: train_images, self.labels: train_labels})
                # 算完一轮后，用现在的时间减去刚刚记录的时间，则计算出一次迭代需要耗费的时间
                duration = time.time() - start_time

                # 做一个每过100轮执行的处理
                if i % 100 == 0:
                    # 每100轮计算一次算力，即每秒能算几轮
                    power = 100 / duration
                    # 把duration浮点化
                    batch_prop_v = float(duration)
                    print("round {}, loss: {:.2f} ({:.1f} examples/sec; {:.3f} sec/batch)".format(i, loss_value, power, batch_prop_v))

            # 用测试集进行正确率的测试
            for i in range(10000 / self.batch_size):
                test_images, test_labels = sess.run([self.test_images, self.test_labels])
                predictions = sess.run([top], feed_dict={self.images: test_images, self.labels: test_labels})
                true_count += np.sum(predictions)

                # 打印正确率信息
            print(f'正确率 = {(true_count / 10000) * 100}')


if __name__ == '__main__':
    cnn = Performance(epochs=10000, batch_size=100)
    cnn.process()




