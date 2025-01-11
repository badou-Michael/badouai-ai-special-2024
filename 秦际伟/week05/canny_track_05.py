# encoding=gbk

import cv2
import numpy as np 

def canny_edge_detection(gray, low_threshold, ratio, kernel_size):
    """
    执行Canny边缘检测
    :param gray: 灰度图像
    :param low_threshold: 低阈值
    :param ratio: 高阈值与低阈值的比例
    :param kernel_size: Sobel算子的孔径大小
    :return: 检测到的边缘图像
    """
    detected_edges = cv2.Canny(gray, low_threshold, low_threshold * ratio, apertureSize=kernel_size)
    return detected_edges

def apply_bitwise_and(img, detected_edges):
    """
    将原始图像与检测到的边缘进行按位与操作
    :param img: 原始图像
    :param detected_edges: 检测到的边缘图像
    :return: 最终的边缘图像
    """
    dst = cv2.bitwise_and(img, img, mask=detected_edges)
    return dst

def display_image(window_name, image):
    """
    显示图像
    :param window_name: 窗口名称
    :param image: 要显示的图像
    """
    cv2.imshow(window_name, image)

def CannyThreshold(lowThreshold):
    detected_edges = canny_edge_detection(gray, lowThreshold, ratio, kernel_size)
    dst = apply_bitwise_and(img, detected_edges)
    display_image('canny result', dst)

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

# 读取图像并转换为灰度图
img = cv2.imread('lenna.jpg')
if img is None:
    print("Error: 图像读取失败")
    exit(1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 创建窗口和滑动条
cv2.namedWindow('canny result')
cv2.createTrackbar('Min threshold', 'canny result', lowThreshold, max_lowThreshold, CannyThreshold)

# 初始化 CannyThreshold 函数
CannyThreshold(0)

# 等待用户按键
cv2.waitKey(0)

# 关闭所有窗口
cv2.destroyAllWindows()
