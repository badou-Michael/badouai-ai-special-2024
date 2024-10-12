from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 灰度化图像
original_image = cv2.imread("lenna.png")
height, width = original_image.shape[:2]  # 获取图像的高宽
grayscale_image = np.zeros((height, width), dtype=original_image.dtype)  # 创建与原图大小相同的单通道图像

# 手动灰度化：BGR转灰度
for row in range(height):
    for col in range(width):
        pixel = original_image[row, col]
        # 转换为灰度值，采用经典的加权法（蓝0.11, 绿0.59, 红0.3）
        grayscale_image[row, col] = int(pixel[0] * 0.11 + pixel[1] * 0.59 + pixel[2] * 0.3)

# 打印最后处理的像素值和灰度图像数组
print("处理后的像素值：", pixel)
print("灰度图像矩阵：\n", grayscale_image)

# 使用OpenCV显示灰度图像
cv2.imshow("手动灰度图像", grayscale_image)

# 使用Matplotlib显示原图
plt.subplot(221)
original_image = plt.imread("lenna.png")
plt.imshow(original_image)
print("---显示原始图像---")
print(original_image)

# 使用skimage库的rgb2gray函数进行灰度化
auto_gray_image = rgb2gray(original_image)  # 转换为灰度图
plt.subplot(222)
plt.imshow(auto_gray_image, cmap='gray')
print("---自动灰度图像---")
print(auto_gray_image)

# 自定义二值化阈值
binary_threshold = 0.5
binary_image = np.where(auto_gray_image >= binary_threshold, 1, 0)  # 将灰度图像转为二值图像
print("---二值化图像---")
print(binary_image)
print("二值化图像形状：", binary_image.shape)

# 显示二值化图像
plt.subplot(223)
plt.imshow(binary_image, cmap='gray')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
