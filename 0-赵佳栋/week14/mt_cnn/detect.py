#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：BadouCV
@File    ：mtcnn.py
@IDE     ：PyCharm
@Author  ：zjd
@Date    ：2025/1/1 13:23
'''
import cv2
import numpy as np
from mtcnn import mtcnn


def main():
    # 读取图像
    img = cv2.imread('img/timg.jpg')
    # 创建 mtcnn 模型实例
    model = mtcnn()
    # 设定三段网络的置信度阈值
    threshold = [0.5, 0.6, 0.7]
    # 人脸检测
    rectangles = model.detectFace(img, threshold)
    # 复制图像用于绘制
    draw = img.copy()

    for rectangle in rectangles:
        if rectangle is not None:
            # 计算矩形框的宽高
            W = -int(rectangle[0]) + int(rectangle[2])
            H = -int(rectangle[1]) + int(rectangle[3])
            # 计算填充量
            paddingH = 0.01 * W
            paddingW = 0.02 * H
            # 裁剪图像
            crop_img = img[int(rectangle[1]+paddingH):int(rectangle[3]-paddingH), int(rectangle[0]-paddingW):int(rectangle[2]+paddingW)]
            if crop_img is None or crop_img.shape[0] < 0 or crop_img.shape[1] < 0:
                continue
            # 绘制矩形框
            cv2.rectangle(draw, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])), (255, 0, 0), 1)
            # 绘制关键点
            for i in range(5, 15, 2):
                cv2.circle(draw, (int(rectangle[i + 0]), int(rectangle[i + 1])), 2, (0, 255, 0), thickness=1)

    # 保存并显示图像
    cv2.imwrite("img/out.jpg", draw)
    cv2.imshow("test", draw)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()