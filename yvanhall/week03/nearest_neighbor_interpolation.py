# -*- coding: utf-8 -*-
import numpy as np
import cv2

# 最邻近插值算法
def nearest_neighbor_interpolation(img):
    # 原图宽高
    height, width, channels = img.shape
    # 把原图放大到 1000 * 1000，创建空矩阵
    # 注意，该算法放大图片不能大于1024，否则img下标会越界
    empty_img = np.zeros((1000, 1000, channels), np.uint8)
    # 宽高缩放倍数
    scale_h = 1000/height
    scale_w = 1000/width

    # 遍历所有像素
    for i in range(1000):
        for j in range(1000):
            # 向下取整
            h = int(i / scale_h + 0.5)
            w = int(j / scale_w + 0.5)
            empty_img[i, j] = img[h, w]
    return empty_img

origin_img = cv2.imread("lenna.png")

# cv2最邻近插值算法
# result_img = cv2.resize(origin_img,(1000, 1000), interpolation=cv2.INTER_NEAREST)

result_img = nearest_neighbor_interpolation(origin_img)
cv2.imshow('nearest.png', result_img)
cv2.waitKey(0)
