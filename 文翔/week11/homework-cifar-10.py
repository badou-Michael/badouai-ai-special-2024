import tensorflow as tf
import numpy as np
import os

# CIFAR-10 数据加载函数
def load_cifar10_batch(filename):
    with open(filename, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
        num_images = data.shape[0] // 3073
        if num_images * 3073 != data.shape[0]:
            raise ValueError(f"The file {filename} does not contain a valid CIFAR-10 batch.")
        labels = data[::3073]  # 每 3073 字节的第一个字节是标签
        images = data.reshape(-1, 3073)[:, 1:]  # 移除每行的标签字节，保留图像数据
        images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)  # 转换为 [batch_size, 32, 32, 3]
    return  images, labels

def load_cifar10_data(data_dir):
    x_train, y_train = [], []
    for i in range(1, 6):  # 加载 data_batch_1 ~ data_batch_5
        images, labels = load_cifar10_batch(os.path.join(data_dir, f"data_batch_{i}.bin"))
        x_train.append(images)
        y_train.append(labels)
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)

    x_test, y_test = load_cifar10_batch(os.path.join(data_dir, "test_batch.bin"))
    return (x_train, y_train), (x_test, y_test)

# 数据增强函数
def data_augmentation(images):
    augmented_images = []
    for img in images:
        if np.random.rand() > 0.5:
            img = np.fliplr(img)
        shift = np.random.randint(-3, 3, size=2)
        img = np.roll(img, shift, axis=(0, 1))
        augmented_images.append(img)
    return np.array(augmented_images)

# 数据预处理
data_dir = r'/cv/第十一周-CNN/cifar/cifar/cifar_data/cifar-10-batches-bin'  # 请根据实际路径修改
(x_train, y_train), (x_test, y_test) = load_cifar10_data(data_dir)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

y_train = tf.one_hot(y_train, depth=10).eval(session=tf.Session())
y_test = tf.one_hot(y_test, depth=10).eval(session=tf.Session())

# 构建卷积神经网络
def cnn_model(x):
    # conv2d：四维张量，卷积通道数，卷积核大小，激活函数
    conv1 = tf.layers.conv2d(x, filters=32, kernel_size=3, activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=2, strides=2)

    conv2 = tf.layers.conv2d(pool1, filters=64, kernel_size=3, activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=2, strides=2)

    # 展平，全连接
    flat = tf.layers.flatten(pool2)

    # 构建全连接层
    dense1 = tf.layers.dense(flat, units=128, activation=tf.nn.relu)
    dense2 = tf.layers.dense(dense1, units=64, activation=tf.nn.relu)
    logits = tf.layers.dense(dense2, units=10)

    return logits

# 输入占位符
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y = tf.placeholder(tf.float32, shape=[None, 10], name='y')

# 构建模型
logits = cnn_model(x)
predictions = tf.nn.softmax(logits)

# 损失函数和优化器
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# 准确率计算
correct_pred = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 超参数
batch_size = 64
epochs = 20

# 会话执行
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]
            batch_x = data_augmentation(batch_x)

            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})

        # 每个epoch计算验证集准确率
        val_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
        print(f"Epoch {epoch + 1}/{epochs}, Validation Accuracy: {val_acc:.4f}")

    # 测试集评估
    test_acc = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
    print(f"Test Accuracy: {test_acc * 100:.2f}%")


