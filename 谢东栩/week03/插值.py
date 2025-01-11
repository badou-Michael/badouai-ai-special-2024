import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取原始图像
img = cv2.imread('cs.jpg')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 设置缩放比例
scale_factor = 2.0

# 原图尺寸
height, width = img_gray.shape

# 使用左上角坐标进行插值
res_nearest_left = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
res_linear_left = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# 使用中心坐标进行插值（通过将坐标加上0.5，模拟中心对齐）
center_x = np.arange(0, width, dtype=np.float32)  # 将中心坐标转换为float32
center_y = np.arange(0, height, dtype=np.float32)  # 将中心坐标转换为float32
center_x += 0.5  # 左上角到中心的偏移量
center_y += 0.5  # 左上角到中心的偏移量

# 使用中心坐标进行插值
res_nearest_center = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
res_linear_center = cv2.resize(img_gray, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

# 设置 Matplotlib 显示中文标签的字体
plt.rcParams['font.sans-serif'] = ['SimHei']

# 绘制结果进行比较
plt.figure(figsize=(10, 8))  # 调整画布大小

# 原图
plt.subplot(2, 3, 1)
plt.imshow(img_gray, cmap='gray')
plt.title('原始图像')
plt.xticks([]), plt.yticks([])

# 使用左上角坐标的最临近插值
plt.subplot(2, 3, 2)
plt.imshow(res_nearest_left, cmap='gray')
plt.title('最临近插值（左上角）')
plt.xticks([]), plt.yticks([])

# 使用左上角坐标的双线性插值
plt.subplot(2, 3, 3)
plt.imshow(res_linear_left, cmap='gray')
plt.title('双线性插值（左上角）')
plt.xticks([]), plt.yticks([])

# 使用中心坐标的最临近插值
plt.subplot(2, 3, 4)
plt.imshow(res_nearest_center, cmap='gray')
plt.title('最临近插值（中心 + 0.5）')
plt.xticks([]), plt.yticks([])

# 使用中心坐标的双线性插值
plt.subplot(2, 3, 5)
plt.imshow(res_linear_center, cmap='gray')
plt.title('双线性插值（中心 + 0.5）')
plt.xticks([]), plt.yticks([])

plt.tight_layout()
plt.show()
