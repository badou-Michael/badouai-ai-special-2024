"""
@author: 赵冬
@since: 2024年9月26日
"""

import cv2
import numpy as np


# 最临近插值
def nearest(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=img.dtype)
    scale_h = src_h / dst_h
    scale_w = src_w / dst_w
    for dst_x in range(dst_w):
        for dst_y in range(dst_h):
            src_x = int(dst_x * scale_w + 0.5)
            src_y = int(dst_y * scale_h + 0.5)
            dst_img[dst_y, dst_x] = img[src_y, src_x]
    return dst_img


# 双线性插值
def bilinear(img, out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim
    if src_h == dst_h and src_w == dst_w:
        return img
    dst_img = np.zeros((dst_h, dst_w, channel), dtype=img.dtype)
    scale_h = src_h / dst_h
    scale_w = src_w / dst_w
    for dst_x in range(dst_w):
        for dst_y in range(dst_h):
            src_x = (dst_x + 0.5) * scale_w - 0.5
            src_y = (dst_y + 0.5) * scale_h - 0.5

            x_0 = int(src_x)
            x_1 = min(x_0 + 1, src_w - 1)

            y_0 = int(src_y)
            y_1 = min(y_0 + 1, src_h - 1)

            temp0 = (x_1 - src_x) * img[y_0, x_0] + (src_x - x_0) * img[y_0, x_1]
            temp1 = (x_1 - src_x) * img[y_1, x_0] + (src_x - x_0) * img[y_1, x_1]

            dst_img[dst_y, dst_x] = (y_1 - src_y) * temp0 + (src_y - y_0) * temp1
    return dst_img


if __name__ == '__main__':
    in_img = cv2.imread("lenna.png")
    out_img1 = nearest(in_img, (800, 800))
    out_img2 = bilinear(in_img, (800, 800))
    cv2.imshow("lenna", np.hstack([out_img1, out_img2]))
    cv2.waitKey(0)
