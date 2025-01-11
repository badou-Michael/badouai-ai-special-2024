from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 读取图片
img = cv2.imread("lenna.png")
h, w = img.shape[:2]                               # 获取图片的高度和宽度
img_gray = np.zeros([h, w], img.dtype)             # 创建一张和当前图片大小一样的单通道图片

# 手动灰度化
for i in range(h):
    for j in range(w):
        m = img[i, j]                              # 取出当前像素的 BGR 值
        img_gray[i, j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)   # 将 BGR 转换为灰度值

# 保存手动生成的灰度图片
cv2.imwrite("gray_image_manual.png", img_gray)

# 使用 OpenCV 进行二值化处理 (thresholding)
threshold_value = 128
_, img_binary = cv2.threshold(img_gray, threshold_value, 255, cv2.THRESH_BINARY)

# 显示二值化后的图像
cv2.imshow("Binary Image", img_binary)
cv2.waitKey(0)  # 按任意键关闭窗口
cv2.destroyAllWindows()

# 保存二值化图像
cv2.imwrite("binary_image.png", img_binary)

# 使用 rgb2gray 进行灰度化
plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray, cmap='gray')

# 保存使用 rgb2gray 生成的灰度图像
plt.imsave("gray_image_rgb2gray.png", img_gray, cmap='gray')

# 二值化处理
img_binary_rgb2gray = img_gray > 0.5  # 使用 0.5 作为阈值，转换为二值图像

# 显示二值化图像
plt.subplot(223)
plt.imshow(img_binary_rgb2gray, cmap='gray')

# 保存二值化图像
plt.imsave("binary_image_rgb2gray.png", img_binary_rgb2gray, cmap='gray')

print("手动灰度图像已保存为 'gray_image_manual.png'")
print("手动二值化图像已保存为 'binary_image.png'")
print("rgb2gray 生成的灰度图像已保存为 'gray_image_rgb2gray.png'")
print("rgb2gray 生成的二值化图像已保存为 'binary_image_rgb2gray.png'")
