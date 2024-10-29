import matplotlib.pyplot as plt
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 加载 MNIST 数据集
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 显示某个数字的函数
def display_digit(image_data, index):
    plt.imshow(image_data[index], cmap=plt.cm.binary)
    plt.title(f"第 {index} 个数字")
    plt.show()

# 显示测试集中第一个数字
display_digit(test_images, 0)

# 构建神经网络模型
def build_model():
    model = models.Sequential([
        layers.Dense(512, activation='relu', input_shape=(28 * 28,)),  # 隐藏层
        layers.Dense(10, activation='softmax')  # 输出层
    ])
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

network = build_model()

# 图像预处理函数：将图像展平并归一化
def preprocess_images(images):
    images = images.reshape((images.shape[0], 28 * 28))  # 将图像展平成一维
    images = images.astype('float32') / 255  # 归一化像素值
    return images

train_images = preprocess_images(train_images)  # 训练数据预处理
test_images = preprocess_images(test_images)    # 测试数据预处理

# 将标签进行 One-hot 编码
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 训练模型
network.fit(train_images, train_labels, epochs=5, batch_size=128)

# 在测试数据集上评估模型
test_loss, test_acc = network.evaluate(test_images, test_labels)
print(f"测试损失: {test_loss:.4f}")
print(f"测试准确率: {test_acc:.4f}")

# 预测某个测试图像的标签
def predict_digit(image_index):
    res = network.predict(test_images)
    predicted_label = res[image_index].argmax()  # 找到概率最大的索引
    print(f"第 {image_index} 个图像的预测结果是: {predicted_label}")

# 显示并预测第二个测试图像
display_digit(test_images.reshape((10000, 28, 28)), 1)  # 将测试数据重塑回二维以便显示，注意这里是单张二维
predict_digit(1)
