# 该文件的目的是构造神经网络的整体结构，并进行训练和测试（评估）过程
import time
import numpy as np
import tensorflow as tf
import cifar10_data_tf2

# 启用 XLA 编译器
max_steps = 4000
batch_size = 100
num_examples_for_eval = 10000
data_dir = "./cifar-10-batches-bin"


# 创建一个variable_with_weight_loss()函数，该函数的作用是：
#   1.使用参数w1控制L2 loss的大小
#   2.使用函数tf.nn.l2_loss()计算权重L2 loss
#   3.使用函数tf.multiply()计算权重L2 loss与w1的乘积，并赋值给weights_loss
#   4.使用函数tf.add_to_collection()将最终的结果放在名为losses的集合里面，方便后面计算神经网络的总体loss，
def variable_with_weight_loss(shape, stddev, w1):
    var = tf.Variable(tf.truncated_normal(shape, stddev=stddev))
    if w1 is not None:
        weights_loss = tf.multiply(tf.nn.l2_loss(var), w1, name='weight_loss')
        tf.add_to_collection('losses', weights_loss)
    return var


# 使用上一个文件里面已经定义好的文件序列读取函数读取训练数据文件和测试数据从文件.
# 其中训练数据文件进行数据增强处理，测试数据文件不进行数据增强处理
# images_train, labels_train = cifar10_data_tf2.pre_process_images(data_dir, batch_size, True)
# images_test, labels_test = cifar10_data_tf2.pre_process_images(data_dir, batch_size, None)
# 加载 CIFAR-10 数据集
(images_train, labels_train), (images_test, labels_test) = tf.keras.datasets.cifar10.load_data()

# 创建tf.data.Dataset对象
train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
train_dataset = train_dataset.shuffle(buffer_size=10000).batch(batch_size)
train_dataset = train_dataset.take(100)

test_dataset = tf.data.Dataset.from_tensor_slices((images_test, labels_test))
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.shuffle(100)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(384, activation='relu'),
    tf.keras.layers.Dense(192, activation='relu'),
    tf.keras.layers.Dense(10)
])

# 添加L2正则化
regularizer = tf.keras.regularizers.l2(0.004)
for layer in model.layers:
    if isinstance(layer, tf.keras.layers.Dense):
        layer.kernel_regularizer = regularizer

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型
start_time = time.time()
for images_batch, labels_batch in train_dataset:
    loss = model.train_on_batch(images_batch, labels_batch)
    duration = time.time() - start_time
    examples_per_second = batch_size / duration
    second_per_batch = float(duration)
    print(f"loss = {loss}, examples/sec = {examples_per_second}")
    start_time = time.time()

# for step in range(max_steps):
#     for images_batch, labels_batch in train_dataset:
#         loss = model.train_on_batch(images_batch, labels_batch)
#         if step % 100 == 0:
#             duration = time.time() - start_time
#             examples_per_second = batch_size / duration
#             second_per_batch = float(duration)
#             print(f"step = {step}，loss = {loss}, examples/sec = {examples_per_second}")
#             start_time = time.time()

# 评估模型
true_count = 0
# total_sample_count = num_examples_for_eval
total_count = 0
for images_batch, labels_batch in test_dataset:
    # 进行预测
    predictions = model.predict(images_batch)
    # 计算当前批次的正确预测数量
    correct_predictions = np.sum(np.argmax(predictions, axis=1) == labels_batch.numpy().flatten())

    # 累加正确预测数量和总样本数量
    true_count += correct_predictions
    total_count += images_batch.shape[0]
    print(f"true_count = {true_count}，total_count = {total_count}")

# 计算准确率
accuracy = true_count / total_count
print(f"Accuracy: {accuracy * 100:.2f}%")
# print(f"accuracy = {true_count / total_count * 100}")
