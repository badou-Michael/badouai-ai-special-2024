# -*- coding: utf-8 -*-
# time: 2024/10/26 16:12
# file: Canny_Track.py
# author: flame
import cv2

# CannyThreshold函数用于应用Canny边缘检测算法，并展示边缘检测的结果
# 参数lowThreshold是Canny算法的低阈值，用于控制边缘检测的敏感度
# 参数highThreshold是Canny算法的高阈值，用于控制边缘检测的敏感度
def CannyThreshold(lowThreshold, highThreshold):
    # 使用Canny算法进行边缘检测
    detected_edges = cv2.Canny(img_gray, lowThreshold, highThreshold, kernel_size)
    # 将检测到的边缘与原图进行位运算，以显示边缘检测的结果
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    # 显示Canny边缘检测的结果
    cv2.imshow("Canny result", dst)

# 初始化低阈值为0
lowThreshold = 0
# 初始化高阈值为100
highThreshold = 100
# 设置高阈值的最大值为100，用于界面中的调节杠
maxThreshold = 100
# 阈值比率，用于计算Canny算法的高阈值
ratio = 3
# Canny算法使用的Sobel算子的大小
kernel_size = 3

# 读取图像文件
img = cv2.imread("lenna.png")
# 将图像转换为灰度图，以便进行边缘检测
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 创建一个窗口用于展示边缘检测的结果
cv2.namedWindow("Canny result")

# 设置低阈值调节杠，允许用户调整Canny算法的低阈值
cv2.createTrackbar("Min threshold", "Canny result", lowThreshold, maxThreshold, lambda x: CannyThreshold(x, highThreshold))
# 设置高阈值调节杠，允许用户调整Canny算法的高阈值
cv2.createTrackbar("Max threshold", "Canny result", highThreshold, maxThreshold, lambda x: CannyThreshold(lowThreshold, x))

# 初始调用CannyThreshold函数，设置低阈值为0，高阈值为100
CannyThreshold(lowThreshold, highThreshold)

# 等待用户操作，如果按下ESC键，则关闭所有窗口
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
