import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('input.jpg')

# 定义源点和目标点
# 这里假设你已经知道图像中要进行透视变换的四个顶点
src_points = np.float32([[100, 100], [200, 100], [100, 200], [200, 200]])
dst_points = np.float32([[50, 50], [250, 50], [50, 250], [250, 250]])

# 计算透视变换矩阵
M = cv2.getPerspectiveTransform(src_points, dst_points)

# 应用透视变换
transformed_image = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))

# 显示原图和变换后的图像
plt.subplot(121), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(122), plt.imshow(cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)), plt.title('Transformed Image')
plt.show()
