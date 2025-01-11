import cv2  # 图像处理。
import numpy as np  # 数值计算和数组操作。
from matplotlib import pyplot as plt  # 用于绘图。

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可
'''

# 获取灰度图像
img = cv2.imread("lenna.png", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 将彩色图像转换为灰度图像。


dst = cv2.equalizeHist(gray)  # 对灰度图像进行直方图均衡化。

# 直方图
hist = cv2.calcHist([dst], [0], None, [256], [0, 256])  # 计算均衡化后图像的直方图。

'''
cv2.calcHist(images,channels,mask,histSize,ranges)

images: 当传入函数时应 用中括号 [] 括来例如[img]
channels: 如果传入图像是灰度图，它的值就是[0]。如果是彩色图像，它的传入的参数可以是[0][1][2]，它们分别对应着BGR。
mask: 掩模图像。统整幅图像的直方图就把它为 None。
histSize:BIN 的数目。
ranges: 像素值范围常为 [0 256]
'''

plt.figure()  # 创建新的图形窗口。
plt.hist(dst.ravel(), 256)  # 绘制均衡化图像的直方图。
plt.show()  # 显示直方图。

cv2.imshow("Histogram_Equalization", np.hstack([gray, dst]))  # 显示原始和均衡化后的图像并排。
cv2.waitKey(0)  # 等待用户按键以关闭窗口。


# 彩色图像直方图均衡化
img = cv2.imread("lenna.png", 1)  # 读取彩色图像。
cv2.imshow("src", img)  # 显示原始彩色图像。

# 彩色图像均衡化,需要分解通道对每一个通道均衡化
(b, g, r) = cv2.split(img)  # 分离彩色图像的三个通道。
bH = cv2.equalizeHist(b)  # 对蓝色通道进行直方图均衡化。
gH = cv2.equalizeHist(g)  # 对绿色通道进行直方图均衡化。
rH = cv2.equalizeHist(r)  # 对红色通道进行直方图均衡化。

# 合并每一个通道
result = cv2.merge((bH, gH, rH))  # 合并均衡化后的三个通道。
cv2.imshow("Equalization_RGB", result)  # 显示均衡化后的彩色图像。

cv2.waitKey(0)  # 等待用户按键以关闭窗口。
