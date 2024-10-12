# -*- coding: utf-8 -*-

import cv2
import numpy as np

# 双线性插值
img = cv2.imread("lenna.png")
img_linear = cv2.resize(img,(2000,2000),interpolation=cv2.INTER_LINEAR)
cv2.imshow("linear",img_linear)
cv2.waitKey(0)

#
# def bilinear_interpolation(img, out_dim):
#     src_h, src_w, channel = img.shape
#     dst_h, dst_w = out_dim[1], out_dim[0]
#
#     if src_h == dst_h and src_w == dst_w:
#         return img.copy()
#
#     dst_img = np.zeros((dst_h, dst_w, 3), dtype=np.uint8)
#     scale_x, scale_y = float(src_w) / dst_w, float(src_h) / dst_h
#     for i in range(3):
#         for dst_y in range(dst_h):
#             for dst_x in range(dst_w):
#                 # 实现中心对称
#                 src_x = (dst_x + 0.5) * scale_x - 0.5
#                 src_y = (dst_y + 0.5) * scale_y - 0.5
#
#                 # 找到原坐标的四个点
#                 src_x0 = int(np.floor(src_x))
#                 src_x1 = min(src_x0 + 1, src_w - 1)
#                 src_y0 = int(np.floor(src_y))
#                 src_y1 = min(src_y0 + 1, src_h - 1)
#
#                 # 代入公式计算
#                 temp0 = (src_x1 - src_x) * img[src_y0, src_x0, i] + (src_x - src_x0) * img[src_y0, src_x1, i]
#                 temp1 = (src_x1 - src_x) * img[src_y1, src_x0, i] + (src_x - src_x0) * img[src_y1, src_x1, i]
#                 dst_img[dst_y, dst_x, i] = int((src_y1 - src_y) * temp0 + (src_y - src_y0) * temp1)
#
#     return dst_img
#
#
# img = cv2.imread('lenna.png')
# dst = bilinear_interpolation(img, (800, 800))
# cv2.imshow('bilinear interp', dst)
# cv2.waitKey()
#
