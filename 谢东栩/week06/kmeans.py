import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取灰度图像,cv2.IMREAD_GRAYSCALE表示0，转换为灰度
img = cv2.imread('cs.jpg', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if img is None:
    raise FileNotFoundError("图像未找到，请检查文件路径是否正确！")

# 获取图像高度和宽度
rows, cols = img.shape

# 将图像数据（二维数组）展平为一维数组并转换为浮点类型，flatten展开数组,astype转换类型
data = img.flatten().astype(np.float32)

# 定义 K-Means 聚类的停止条件和参数，迭代次数最多10或者中心聚类误差小于1.0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

# 执行 K-Means 聚类，将图像分为 4 类
compactness, labels, centers = cv2.kmeans(data, 2, None, criteria, 10, flags)

# 将标签重新调整为与原图相同的形状
dst = labels.reshape(img.shape)

# 设置 Matplotlib 显示中文标签的字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 显示原始图像和聚类结果
titles = ['原始图像', '聚类图像']
images = [img, dst]

plt.figure(figsize=(10, 5))  # 调整窗口大小
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    #plt.axis('off')  # 隐藏坐标轴
plt.tight_layout()
plt.show()
