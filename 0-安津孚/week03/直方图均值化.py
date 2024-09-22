import cv2
import numpy as np
from matplotlib import pyplot as plt

# 灰度图像直方图
# 获取灰度图像
img = cv2.imread("../week02/lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("image_gray", gray)

# 灰度图像的直方图，方法一
plt.figure()
plt.hist(gray.ravel(), 256)
plt.title('Histogram')
plt.xlabel('Pixel value')
plt.ylabel('Frequency')
plt.show()

'''
#彩色图像直方图
'''
image = cv2.imread("../week02/lenna.png")
# cv2.waitKey(0)

chans = cv2.split(image)
colors = ("b", "g", "r")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for (chan, color) in zip(chans, colors):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=color)
    plt.xlim([0, 256])
plt.show()


# 读取图像
img = cv2.imread("../week02/lenna.png", 1)

# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰度图像直方图均衡化
dst = cv2.equalizeHist(gray)

# 计算直方图均衡化后的图像dst的直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])

# 创建一个新的图形窗口
plt.figure()

# 绘制图像dst的直方图
plt.hist(dst.ravel(), 256, [0, 256])
plt.title("Histogram for Equalized Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# 将原始灰度图像和均衡化后的图像并排显示
cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))

cv2.waitKey(0)
cv2.destroyAllWindows()

'''
# 彩色图像直方图均衡化
'''


# 读取图像
img = cv2.imread("../week02/lenna.png", 1)
# cv2.imshow("src", img)

# 彩色图像均衡化,需要分解通道 对每一个通道均衡化
(b, g, r) = cv2.split(img)
bH = cv2.equalizeHist(b)
gH = cv2.equalizeHist(g)
rH = cv2.equalizeHist(r)
# 合并每一个通道
result = cv2.merge((bH, gH, rH))

# 使用matplotlib显示图像
plt.figure(figsize=(10, 5))  # 设置图像大小
plt.subplot(1, 2, 1)  # 1行2列的第1个位置
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # 将BGR转换为RGB
plt.title('Original Image')

plt.subplot(1, 2, 2)  # 1行2列的第2个位置
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))  # 将BGR转换为RGB
plt.title('Equalized Image')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

