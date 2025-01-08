import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# 数据加载
def load_data(file_path):
    with open(file_path, 'r') as f:
        data_list = f.readlines()
    return data_list

train_data_list = load_data("dataset/mnist_train.csv")
test_data_list = load_data("dataset/mnist_test.csv")

# 数据预处理
def preprocess_data(data_list):
    inputs = []
    labels = []
    for record in data_list:
        all_values = record.split(',')
        inputs.append((np.asfarray(all_values[1:]) / 255.0).tolist())
        labels.append(int(all_values[0]))
    return np.array(inputs), np.array(labels)

train_inputs, train_labels = preprocess_data(train_data_list)
test_inputs, test_labels = preprocess_data(test_data_list)

# 构建模型
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(200, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 训练模型
model.fit(train_inputs, train_labels, epochs=10, batch_size=32)

# 测试模型
test_loss, test_accuracy = model.evaluate(test_inputs, test_labels)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# 随机选择一张图片展示并预测
random_index = random.randint(0, len(test_inputs) - 1)
random_image = test_inputs[random_index].reshape(28, 28)
plt.imshow(random_image, cmap='Greys', interpolation='None')
plt.title(f"Correct Label: {test_labels[random_index]}")
plt.show()

# 对随机图片进行预测
random_input = test_inputs[random_index].reshape(1, 784)
predicted_label = np.argmax(model.predict(random_input))
print(f"Predicted Label: {predicted_label}")
