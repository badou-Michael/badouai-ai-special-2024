import cv2
import numpy as np
import random
import matplotlib.pyplot as plt

# 读取图像
img1 = cv2.imread('iphone1.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('iphone2.png', cv2.IMREAD_GRAYSCALE)

# 初始化SIFT检测器
sift = cv2.SIFT_create()

# 提取图像的特征点
keypoints1, _ = sift.detectAndCompute(img1, None)
keypoints2, _ = sift.detectAndCompute(img2, None)

# 合并特征点数据
data_points = np.array([kp.pt for kp in keypoints1 + keypoints2])

# RANSAC参数
iterations = 100
threshold = 5
best_inliers = 0
best_model = (0, 0)

# RANSAC迭代过程
for _ in range(iterations):
    # 随机选择两个点来拟合直线
    sample_indices = random.sample(range(len(data_points)), 2)
    (x1, y1), (x2, y2) = data_points[sample_indices]

    # 避免除以零错误
    if x2 - x1 != 0:
        a = (y2 - y1) / (x2 - x1)
        b = y1 - a * x1
    else:
        continue

    # 计算模型内点
    inliers = 0
    for (x, y) in data_points:
        y_pred = a * x + b
        if abs(y - y_pred) < threshold:
            inliers += 1

    # 更新最佳模型
    if inliers > best_inliers:
        best_inliers = inliers
        best_model = (a, b)

# 绘制RANSAC结果
a, b = best_model
plt.scatter(data_points[:, 0], data_points[:, 1], label='Key Points')
x_vals = np.linspace(data_points[:, 0].min(), data_points[:, 0].max(), 100)
plt.plot(x_vals, a * x_vals + b, color='red', label='RANSAC Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('RANSAC Line Fitting on Key Points')
plt.show()
