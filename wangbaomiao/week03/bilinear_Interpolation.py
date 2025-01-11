# -*- coding: utf-8 -*-
# time: 2024/10/25 18:18
# file: bilinear_Interpolation.py
# author: flame
import numpy as np
import cv2
def bilinear_interpolation(img, out_dim):
    """
    使用双线性插值法对图像进行缩放。

    Parameters:
    img: ndarray - 输入图像，以NumPy数组形式表示。
    out_dim: tuple - 输出图像的目标尺寸，格式为 (宽度, 高度)。

    Returns:
    ndarray - 缩放后的图像，以NumPy数组形式表示。
    """
    # 获取源图像的高度、宽度和通道数
    src_h, src_w, channel = img.shape
    # 提取目标尺寸的高度和宽度
    dst_h, dst_w = out_dim[1], out_dim[0]

    # 打印源图像和目标图像的尺寸，用于调试
    print("src_h, src_w = ", src_h, src_w)
    print("dst_h, dst_w = ", dst_h, dst_w)

    # 如果源图像和目标图像的尺寸相同，直接返回源图像的副本
    if src_h == dst_h and src_w == dst_w:
        return img.copy()

    # 创建一个与目标尺寸相同且通道数为3的零数组，用于存储缩放后的图像
    dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)

    # 计算x方向和y方向的缩放比例
    scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h

    # 遍历每个通道
    for i in range(channel):
        # 遍历目标图像的每个像素
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                # 计算目标像素在源图像中的对应位置
                # 使用几何中心对称的方法计算源图像的坐标
                # 直接计算方法为：src_x = dst_x * scale_x
                src_x = (dst_x + 0.5) * scale_x - 0.5
                src_y = (dst_y + 0.5) * scale_y - 0.5

                # 确定用于插值的整数坐标
                src_x0 = int(np.floor(src_x))  # np.floor() 返回不大于输入参数的最大整数（向下取整）
                src_x1 = min(src_x0 + 1, src_w - 1)
                src_y0 = int(np.floor(src_y))
                src_y1 = min(src_y0 + 1, src_h - 1)

                # 进行双线性插值
                # 计算水平方向的插值
                temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
                temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
                # 计算垂直方向的插值并赋值给目标图像
                dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)

    # 返回缩放后的图像
    return dst_img


# 当作为主程序运行时，读取 'lenna.png' 图像，使用双线性插值法将其缩放到 700x700，并显示结果
if __name__ == '__main__':
    img = cv2.imread('lenna.png')  # 读取图像
    dst = bilinear_interpolation(img, (700, 700))  # 调用双线性插值函数进行缩放
    cv2.imshow('bilinear interp', dst)  # 显示缩放后的图像
    cv2.waitKey()  # 等待用户按键关闭窗口
