#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author : chairDu
# @Email : chair7@163.com
# @File : day2.py
# @DataTime : 2024-09-17 21:52:00
# @Description ：图片写入本地`



import cv2
from matplotlib import pyplot as plt
import math
#  保存单张图片
def save_image_one(binary_img, file_name):
    """
    :param binary_img: 图片
    :param file_name: 文件名字
    :return:打印图片以及保存
    """
    cv2.imshow("灰度化图片", binary_img)  # 显示灰度化图片 
    cv2.waitKey(0)  # 等待2秒
    cv2.destroyAllWindows()  # 关闭图片
    file_name = f"./jpg/{file_name}.png"
    cv2.imwrite(f"{file_name}", binary_img)
    print(f"图片已保存到{file_name}")
    
# 保存缩略图
def save_image_thumbnail(img_list, file_name, img_type = 1):
    """ 图片放到一张图,存入本地
    :param img_list: 图片列表
    :param file_name: 文件名字
    :return:打印图片
    """
    if img_type != 1:
        img_list = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in img_list]   # bgr转rgb
    # if img_type == 1: #rgb转bgr
    #     img_list = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in img_list]
    n = math.ceil(len(img_list) ** 0.5)
    for i in range(len(img_list)):
        plt.subplot(n, n, i + 1)    # 子图
        plt.axis('off') # 关闭坐标轴
        # 消除图片边距
        plt.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)    # 关闭坐标轴)
        plt.imshow(img_list[i])
    # 图片写入本地
    file_name = f"./jpg/{file_name}-{len(img_list)}张"
    print(file_name)
    plt.savefig(f"{file_name}.png")
    plt.show()
