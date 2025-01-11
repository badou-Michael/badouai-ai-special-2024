# -*- coding: utf-8 -*-
# time: 2024/10/16 19:40
# file: nearest_interp.py.py
# author: flame

# 导入必要的库
import cv2
import numpy as np

def function(img):
    """
    使用最近邻插值法对图像进行放大。

    参数:
    img: 待处理的原始图像，是一个三维数组，表示图像的每个像素点。

    返回值:
    等于原始图像使用最近邻插值法放大到800x800后的图像。
    """
    # 获取原始图像的高度、宽度和通道数
    height, width, channels = img.shape
    # 创建一个800x800大小的空白图像，用于存放插值后的结果
    emptyImage = np.zeros((800, 800, channels), np.uint8)
    # 计算目标图像与原始图像在高度和宽度上的缩放比例
    sh = 800 / height
    sw = 800 / width

    # 遍历目标图像的每个像素点
    for i in range(800):
        for j in range(800):
            # 根据缩放比例计算出在原始图像中的对应位置，并取整
            x = int(i / sh + 0.5)
            y = int(j / sw + 0.5)
            # 将原始图像中(x, y)位置的像素值赋给目标图像中的(i, j)位置
            emptyImage[i, j] = img[x, y]

    # 返回插值后的图像
    return emptyImage


def function2(img):
    """
    使用最近邻插值法对图像进行放大。

    参数:
    img: 待处理的原始图像，是一个三维数组，表示图像的每个像素点。

    返回值:
    等于原始图像使用最近邻插值法放大到800x800后的图像。
    """
    # 输入验证
    if not isinstance(img, np.ndarray) or img.ndim != 3:
        raise ValueError("输入必须是一个三维数组")

    # 获取原始图像的高度、宽度和通道数
    height, width, channels = img.shape

    # 创建一个800x800大小的空白图像，用于存放插值后的结果
    emptyImage = np.zeros((800, 800, channels), np.uint8)

    # 计算目标图像与原始图像在高度和宽度上的缩放比例
    sh = 800 / height
    sw = 800 / width

    # 创建目标图像的坐标网格
    x_coords = np.arange(800) / sh
    y_coords = np.arange(800) / sw

    # 使用向量化操作计算目标图像的每个像素点对应的原始图像坐标
    x_coords = np.clip(np.round(x_coords).astype(int), 0, height - 1)
    y_coords = np.clip(np.round(y_coords).astype(int), 0, width - 1)

    # 使用广播机制将坐标映射到目标图像
    emptyImage[:, :, :] = img[x_coords[:, None], y_coords[None, :], :]

    # 返回插值后的图像
    return emptyImage

# 读取原始图像
img = cv2.imread("lenna.png")
# 调用函数对图像进行最近邻插值放大
#zoom = function(img)
zoom = cv2.resize(img, (800, 800), interpolation=cv2.INTER_LINEAR)
# 打印插值后的图像信息
print(zoom)
print(zoom.shape)

# 显示插值后的图像和原始图像
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
# 等待用户按键，用于控制窗口关闭
cv2.waitKey(0)
