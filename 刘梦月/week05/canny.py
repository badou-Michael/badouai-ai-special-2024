# 1. 从头实现Canny边缘检测算法

import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('A4.jpg', cv2.IMREAD_GRAYSCALE)

# 1. 高斯模糊来减少噪声
blurred = cv2.GaussianBlur(image, (5, 5), 1.4)

# 2. 计算梯度（Sobel算子）
gradient_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
angle = np.arctan2(gradient_y, gradient_x)

# 3. 非极大值抑制
angle = angle * 180. / np.pi
angle[angle < 0] += 180
nms = np.zeros_like(magnitude)
for i in range(1, magnitude.shape[0] - 1):
    for j in range(1, magnitude.shape[1] - 1):
        try:
            q = 255
            r = 255
            # 0度
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # 45度
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # 90度
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # 135度
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                nms[i, j] = magnitude[i, j]
            else:
                nms[i, j] = 0
        except IndexError as e:
            pass

# 4. 双阈值检测
high_threshold = nms.max() * 0.2
low_threshold = high_threshold * 0.5
res = np.zeros_like(nms)
strong = np.uint8(255)
weak = np.uint8(75)
strong_i, strong_j = np.where(nms >= high_threshold)
zeros_i, zeros_j = np.where(nms < low_threshold)
weak_i, weak_j = np.where((nms <= high_threshold) & (nms >= low_threshold))

res[strong_i, strong_j] = strong
res[weak_i, weak_j] = weak

# 5. 边缘跟踪
for i in range(1, res.shape[0] - 1):
    for j in range(1, res.shape[1] - 1):
        if (res[i, j] == weak):
            if ((res[i + 1, j - 1] == strong) or (res[i + 1, j] == strong) or (res[i + 1, j + 1] == strong)
                    or (res[i, j - 1] == strong) or (res[i, j + 1] == strong)
                    or (res[i - 1, j - 1] == strong) or (res[i - 1, j] == strong) or (res[i - 1, j + 1] == strong)):
                res[i, j] = strong
            else:
                res[i, j] = 0

# 显示结果
plt.figure(figsize=(10, 10))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(res, cmap='gray'), plt.title('Canny Edge Detection')
plt.show()


# 2. cv2.Canny()函数实现Canny边缘检测算法

import cv2
import matplotlib.pyplot as plt

# 读取图像并转换为灰度
image = cv2.imread('A4.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用OpenCV的Canny函数
# 参数1: 低阈值
# 参数2: 高阈值
edges = cv2.Canny(gray, 50, 200)

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(gray, cmap='gray'), plt.title('Original Image')
plt.subplot(122), plt.imshow(edges, cmap='gray'), plt.title('Canny Edge Detection')
plt.show()
