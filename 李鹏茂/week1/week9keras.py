# 导入必要的库
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

# ========================
# [1] 加载 MNIST 数据集
# ========================
# mnist.load_data() 下载并加载 MNIST 数据集
# 该数据集包含60,000张训练图像和10,000张测试图像
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 打印训练集和测试集的形状
print('train_images.shape = ', train_images.shape)  # 训练集图片形状 (60000, 28, 28)
print('test_images.shape = ', test_images.shape)    # 测试集图片形状 (10000, 28, 28)
print('test_labels:', test_labels)                  # 打印测试集标签（手写数字标签）

# ========================
# [2] 显示测试集中的一张图片
# ========================
# 获取测试集中的第一张图片
digit = test_images[0]

# 使用matplotlib显示该图片，cmap=plt.cm.binary表示灰度图
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# ========================
# [3] 搭建神经网络模型
# ========================
# 创建一个顺序模型（Sequential），每一层按顺序连接
network = models.Sequential()

# 添加第一层：全连接层(Dense)，有512个神经元，使用ReLU激活函数，输入数据大小为28*28的图片展平后的一维数组
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))

# 添加第二层：输出层，使用softmax激活函数，输出10个数字的概率（分类为0-9）
network.add(layers.Dense(10, activation='softmax'))

# 编译模型，选择优化器'rmsprop'，损失函数'categorical_crossentropy'，并且评估准确度
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# ========================
# [4] 数据预处理
# ========================
# 将训练图像从28x28二维数组展平为一维数组（每个图像包含28*28=784个像素点）
train_images = train_images.reshape((60000, 28*28))
test_images = test_images.reshape((10000, 28*28))

# 将像素值归一化到0到1之间（原始数据是0到255之间的整数）
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 将标签转换为one-hot编码（例如数字7变成[0,0,0,0,0,0,0,1,0,0]）
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 打印转换前后的标签，以便验证
print("before change:", test_labels[0])
print("after change:", test_labels[0])

# ========================
# [5] 训练神经网络
# ========================
# 训练模型，使用训练集数据，训练5个epoch，每次批处理128张图片
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# ========================
# [6] 在测试集上评估模型
# ========================
# 使用测试集评估模型的性能，计算损失值和准确率
test_loss, test_acc = network.evaluate(test_images, test_labels, verbose=1)

# 打印测试集上的损失值和准确率
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

# ========================
# [7] 用模型进行预测
# ========================
# 从测试集中选择一张图片进行预测
digit = test_images[1]  # 选择第二张图片
plt.imshow(digit, cmap=plt.cm.binary)  # 显示该图片
plt.show()

# 对整个测试集进行预测，得到每个数字的概率分布
res = network.predict(test_images)

# 输出预测结果：打印预测的数字
for i in range(res[1].shape[0]):
    if res[1][i] == 1:  # 找到预测概率为1的数字
        print("该图片的数字是：", i)
        break
