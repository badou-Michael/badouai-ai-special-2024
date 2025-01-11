# # Canny是目前最优秀的边缘检测算法之一，其目标为找到一个最优的边缘，其最优边缘的定义为：
# 1、好的检测：算法能够尽可能的标出图像中的实际边缘
# 2、好的定位：标识出的边缘要与实际图像中的边缘尽可能接近
# 3、最小响应：图像中的边缘只能标记一次

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
主要参数说明
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图像
第二个参数是阈值1
第三个参数是阈值2
'''

import cv2

img = cv2.imread('lenna.png',1)    # 读取图像 mdoe = 1 表示读取彩色图像
# img = cv2.imread('lenna.png',0)    # 读取图像 mode = 0 表示读取灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换彩色图像为灰度图像


low_threshold = 100
high_threshold = 200

# edges = cv2.Canny(gray, low_threshold, high_threshold)
# cv2.imshow('canny', edges)

cv2.imshow('canny', cv2.Canny(gray, low_threshold, high_threshold))  # 显示图像 cv2.Canny()函数进行边缘检测 200, 300 阈值

cv2.waitKey()
cv2.destroyAllWindows()
