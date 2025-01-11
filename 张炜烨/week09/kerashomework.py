import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical

# 加载 MNIST 数据集
def load_data():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    print(f"Train Images Shape: {train_images.shape}")
    print(f"Train Labels: {train_labels[:10]}")
    print(f"Test Images Shape: {test_images.shape}")
    print(f"Test Labels: {test_labels[:10]}")
    return train_images, train_labels, test_images, test_labels

# 显示图片
def display_image(image, title="Handwritten Digit"):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(title)
    plt.axis('off')
    plt.show()

# 数据预处理：归一化和标签转换为 One-Hot 编码
def preprocess_data(train_images, test_images, train_labels, test_labels):
    train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
    test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)
    return train_images, test_images, train_labels, test_labels

# 构建神经网络模型
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 训练模型
def train_model(model, train_images, train_labels, epochs=5, batch_size=128):
    model.fit(train_images, train_labels, epochs=epochs, batch_size=batch_size)

# 测试模型
def evaluate_model(model, test_images, test_labels):
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
    print(f"Test Loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")

# 单张图片预测
def predict_single_image(model, test_images, index):
    digit = test_images[index].reshape(28, 28)  # 恢复为原始图片形状
    display_image(digit, title=f"Test Image at Index {index}")
    test_images_flat = test_images.reshape((10000, 28 * 28))
    prediction = model.predict(test_images_flat)
    predicted_label = np.argmax(prediction[index])
    print(f"The predicted number for the image is: {predicted_label}")

# 主函数
if __name__ == "__main__":
    # 加载数据
    train_images, train_labels, test_images, test_labels = load_data()

    # 显示测试集中的第一张图片
    display_image(test_images[0], title="Sample Test Image (Index 0)")

    # 数据预处理
    train_images, test_images, train_labels, test_labels = preprocess_data(
        train_images, test_images, train_labels, test_labels
    )

    # 构建并训练模型
    model = build_model()
    train_model(model, train_images, train_labels)

    # 测试模型
    evaluate_model(model, test_images, test_labels)

    # 单张图片预测
    predict_single_image(model, test_images, index=1)