#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time  : 2024/11/12 13:59
# @Author: Gift
# @File  : gaussion_blur_pic.py 
# @IDE   : PyCharm
import cv2
def gaussian_blur_image_cv2(input_path, output_path, kernel_size=(5, 5), sigmaX=0):
    """
    使用OpenCV进行高斯模糊
    :param input_path: 输入图片路径
    :param output_path: 输出图片路径
    :param kernel_size: 高斯核大小，默认为(5, 5)
    :param sigmaX: 高斯核的标准差，默认为0
    """
    # 读取图片
    img = cv2.imread(input_path)
    # 进行高斯模糊
    blurred_img = cv2.GaussianBlur(img, kernel_size, sigmaX)
    # 保存图片
    cv2.imwrite(output_path, blurred_img)
if __name__ == '__main__':
    input_path = 'lenna.png'
    output_path = 'lenna_gaussian_blur.png'
    gaussian_blur_image_cv2(input_path, output_path,sigmaX=6)
