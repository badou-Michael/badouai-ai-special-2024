import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
img = cv2.imread('lenna.jpg')

# 检查图像是否成功加载
if img is None:
    print("图像加载失败，请检查路径！")
else:
    # 1. 灰度化处理
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 二值化处理（使用固定阈值）
    _, binary_img = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # 设置 Matplotlib 显示中文标签的字体
    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 显示图像
    cv2.imshow('原始图像', img)
    cv2.imshow('灰度图像', gray_img)
    cv2.imshow('二值图像', binary_img)

    # 等待按键按下后关闭所有窗口
    cv2.waitKey(0)
    cv2.destroyAllWindows()
