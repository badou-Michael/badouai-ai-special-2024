import cv2
import numpy as np
import matplotlib.pyplot as plt

# 彩色直方图均衡化方法一

# 读取名为"lenna.png"的彩色图像，第二个参数1表示以彩色模式读取
img = cv2.imread("lenna.png", 1)
# 显示原始彩色图像，并命名窗口为"src"
cv2.imshow("src", img)

# 彩色图像均衡化，需要分解通道对每一个通道均衡化
# 使用OpenCV的split函数将彩色图像拆分为三个通道（蓝色b、绿色g、红色r）
(b, g, r) = cv2.split(img)
# 对蓝色通道进行直方图均衡化
bH = cv2.equalizeHist(b)
# 对绿色通道进行直方图均衡化
gH = cv2.equalizeHist(g)
# 对红色通道进行直方图均衡化
rH = cv2.equalizeHist(r)
# 合并均衡化后的三个通道
result = cv2.merge((bH, gH, rH))
# 显示均衡化后的彩色图像，并命名窗口为"dst_rgb"
cv2.imshow("dst_rgb", result)

cv2.waitKey(0)


# 彩色直方图均衡化方法二
# def histogram_equalization_color(img):
#     # 将彩色图像从BGR色彩空间转换到YCrCb色彩空间
#     ycrcb = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
#     # 分离YCrCb图像的三个通道
#     y,cr,cb = cv2.split(ycrcb)
#     # 对亮度通道Y进行直方图均衡化
#     y_eq = cv2.equalizeHist(y)
#     # 合并均衡化后的亮度通道和原始的Cr、Cb通道
#     equalized_ycrcb = cv2.merge((y_eq,cr,cb))
#
#     return cv2.cvtColor(equalized_ycrcb,cv2.COLOR_YCrCb2BGR)
#
# img = cv2.imread("lenna.png")
# equalized_img = histogram_equalization_color(img)
#
# cv2.imshow('Original', img)
# cv2.imshow('Equalized', equalized_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#
# # 灰度直方图均衡化
# # 读取名为"lenna.png"的图像
# img = cv2.imread("lenna.png")
# # 将彩色图像转换为灰度图像
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# plt.figure("source1")
# # 绘制原始灰度图像的直方图，gray.ravel()将灰度图像展平为一维数组，256表示将像素值范围分为256个区间
# plt.hist(gray.ravel(), 256)
#
# # 对灰度图像进行直方图均衡化
# equalized = cv2.equalizeHist(gray)
#
# plt.figure("equalized")
# # 绘制均衡化后的灰度图像的直方图
# plt.hist(equalized.ravel(), 256)
#
# cv2.imshow("source", img)
# cv2.imshow("source1", gray)
# cv2.imshow("equalized", equalized)
# plt.show()
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
