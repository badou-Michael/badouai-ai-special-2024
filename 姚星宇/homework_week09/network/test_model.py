from keras.models import load_model
from keras.datasets import mnist
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# 加载模型
model = load_model('mnist_model.keras')
print("Model loaded from mnist_model.keras")

# 加载数据
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
digit = test_images[99]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()

# 准备测试数据
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255
test_labels = to_categorical(test_labels)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# 预测测试数据
res = model.predict(test_images)

# 输出预测结果
# for i in range(10):
#     print(res[99][i])
# 这里不能直接用 res[99][i] == 1来进行判断，因为是浮点数，它的结果不一定正好是1
for i in range(res[99].shape[0]):
    if (res[99][i] > 0.99):
        print("the number of the picture is : ", i)
        break