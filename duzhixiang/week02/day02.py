#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : chairDu
# @Email : chair7@163.com
# @File : day2.py
# @DataTime : 2024-09-04 21:52:00
# @Description ：
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
# from skimage.color import rgb2gray        # 没用到
import numpy as np
# import matplotlib.pyplot as plt   #安装无法适配,一直报找不到
# from PIL import Image # 没用到
import cv2


def graying_picture(img, img_type=1):
    """
    灰度
    :param img: 图片样式
    :param img_type: 1(BGR格式), 2(RGB格式)
    :return: 返回图片gbk
    """
    gbk = [0.114, 0.587, 0.299]
    gbk = gbk if img_type == 1 else gbk[::-1]
    gray_img = np.einsum('ijk,k->ij', img, gbk).astype(np.uint8)
    return gray_img


def binary_picture(img, img_type=1):
    """
    二值化
    :param img: 图片样式
    :param img_type: 1(BGR格式), 2(RGB格式)
    :return: 返回图片gbk
    """
    new_raying_img = graying_picture(img=img, img_type=img_type )
    h, w = new_raying_img.shape
    for _h in range(h):
        for _w in range(w):
            u = new_raying_img[_h, _w]
            if img_type == 1:
                new_raying_img[_h, _w] = 255 if u >= 255 / 2 else 0
            else:
                new_raying_img[_h, _w] = 0 if u >= 255 / 2 else 255
    return new_raying_img


def color_shift(img, img_type):
    """色彩加深"""
    h, w = img.shape[:2]
    for _h in range(h):
        for _w in range(w):
            u = img[_h, _w]
            for i in range(3):
                if img_type == 1:
                #-10%  if gbk > 中间值 else +10%
                    u[i] = u[i] - u[i] * 0.1 if u[i] >= u[1] / 2 else u[i] + u[i] * 0.1
                else:
                    u[i] = u[i] + u[i] * 0.1 if u[i] >= u[1] / 2 else u[i] - u[i] * 0.1
    return img


def display_and_save_image(binary_img, file_name):
    """
    :param binary_img: 图片
    :param file_name: 文件名字
    :return:打印图片以及保存
    """
    cv2.imshow("灰度化图片", binary_img)  # 显示灰度化图片
    cv2.waitKey(0)  # 等待2秒
    cv2.destroyAllWindows()  # 关闭图片
    cv2.imwrite(f"{file_name}.png", binary_img)


def process_image(img_type=1):
    img_gbk = cv2.imread("old.png")  # 默认是gbk格式
    if img_gbk is None:
        return "图片格式存在问题"

    # 灰度化
    graying_img = graying_picture(img=img_gbk, img_type=img_type)
    display_and_save_image(graying_img, file_name="灰度化")

    # 二值化
    graying_img = binary_picture(img=img_gbk, img_type=img_type)
    display_and_save_image(graying_img, "二值化")

    # 二值化
    color_shift_img = color_shift(img=img_gbk, img_type=img_type)
    display_and_save_image(color_shift_img, "色彩加深")


if __name__ == '__main__':
    process_image()
