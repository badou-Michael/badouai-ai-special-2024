#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
@author: Gift

双线性插值 计算量较大，但没有灰度不连续的缺点，图像看起来更光滑
"""
import numpy as np
import cv2
def bilinear_interpolation(img,out_dim):
    """
    双线性插值
    :param img: cv2.imread()返回的类
    :param out_dim:一个元组，(x,y) 目标图像的大小
    :return: dst_img
    """
    #获取原始图像的高，宽，通道数
    src_h,src_w,src_channel = img.shape
    #获取目标图像的高，宽
    dst_h,dst_w = out_dim[0], out_dim[1]
    print(f'src-h={src_h},src_w={src_w}')
    print(f"dst-h={dst_h},dst_w={dst_h}")
    #如果目标图像和原始图像大小相同，则直接复制图片
    if src_h == dst_h and src_w == dst_h:
        return img.copy() #return执行玩退出函数
    #创建一个空的目标图像
    dst_img = np.zeros((dst_h,dst_w,3),np.uint8)
    #缩放的比例
    scale_x = src_w/dst_w
    scale_y = src_h/dst_h
    #遍历目标图像的各个坐标的数据
    for channel in range(src_channel):
        for dst_y in range(dst_h):
            for dst_x in range(dst_w):
                #目标和原始图像几何中心对齐
                #如果不采用中心对其可以直接用 src_x = dst_x * scale_x src_y = dst_y * scale_y
                src_x = (dst_y + 0.5)*scale_x - 0.5
                src_y = (dst_x + 0.5)*scale_y - 0.5
                #找出原图对于点坐标，用于计算插值
                #src_x0 <= src_x <= src_x1
                #src_y0 <= src_y <= src_y1
                #取最靠近目标图片坐标小于1的宽度方向的x值
                src_x0 = int(np.floor(src_x) if np.floor(src_x) < 512 else 511) #返回 不大于输入参数的最大整数，即为向下取整
                src_x1 = min(src_x0 + 1,src_w -1) #减1 是为了边缘化处理，防止宽度方向颜色取出了原始图片的宽度方向的值
                src_y0 = int(np.floor(src_y) if np.floor(src_y) < 512 else 511) #返回 不大于输入参数的最大整数，即为向下取整
                src_y1 = min(src_y0+ 1,src_h - 1) #减1 是为了边缘化处理，防止高度方向颜色取出了原始图片的高度方向的值
                print(f"src_x0={src_x0},src_x1={src_x1},src_x={src_x},src_y={src_y},src_y0={src_y0},src_y1={src_y1}")
                #开始计算插值
                #x方向的插值
                # tmp01 = (x1-x)* Q11 + (x - x0))* Q21 权重
                # tmp02 = (x1-x)* Q12 + (x - x0))* Q22 权重
                tmp01 = (src_x1 - src_x) * img[src_y0,src_x0,channel] + (src_x - src_x0) * img[src_y0,src_x1,channel]
                tmp02 = (src_x1 - src_x) * img[src_y1,src_x0,channel] + (src_x - src_x0) * img[src_y1,src_x1,channel]
                #综合一下
                print(f"dst_x={dst_x} dst_h={dst_y}")
                dst_img[dst_x-1,dst_y,channel] = int((src_y1 - src_y)*tmp01+(src_y - src_y0)*tmp02)
    return dst_img

img = cv2.imread("lenna.png")
dst_image = bilinear_interpolation(img,(700,700))
cv2.imshow('bilinear_interp',dst_image)
cv2.waitKey(0)

