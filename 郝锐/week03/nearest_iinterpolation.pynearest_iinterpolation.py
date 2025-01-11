#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Gift

图像的最近邻插值
用于图像的缩放
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

def nearest_interpolation(image,dst_width,dst_height):
    """
    图像的最邻近插值
    :param image: 原始图像
    :param dst_width: 目标图像的宽度
    :param dst_height: 目标图像的高度
    :return: 目标图像
    """
    # image = cv2.imread(image)
    #获取原始图像的尺寸
    src_width,src_height = image.shape[:2]
    print(src_width,src_height) # 512 512
    #创建目标一个空的目标图像
    dst_image = np.zeros((dst_height,dst_width,3),dtype=np.uint8)
    #遍历新图象中的每个像素点
    for i in range(dst_height):
        for j in range(dst_width):
            # 计算原图象中对应的坐标,加入边缘计算
            src_x = int(i*(src_width / dst_width)+0.5)
            src_y = int(j*(src_height / dst_height)+0.5)
            if src_x < src_width and src_y < src_height: #边缘检测
            #将原图象中对应坐标的像素值赋值给目标图像的对应位置
                dst_image[i,j] = image[src_x,src_y]
            else: #超出边界的填充黑色
                dst_image[i,j] = (0,0,0)
    return dst_image


if __name__ == '__main__':
    #读取图像
    image = cv2.imread("lenna.png")
    image_rgb = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # print(image)
    #图像缩放
    dst_image = nearest_interpolation(image,800,800)
    #显示图像
    cv2.imshow("the original BGR pic",image) #显示CV2读取的BGR图片
    cv2.imshow("the original RBG pic", image_rgb) #显示转换后的RBG图片
    cv2.imshow("resized pic",dst_image) #此处类型依然为BGR
    plt.subplot(121)
    plt.imshow(image,cmap='viridis') #默认显示的是RBG色域的图片
    plt.title('Original Image')
    plt.subplot(122)
    plt.imshow(dst_image)
    plt.title('Resized Image')
    plt.show()
