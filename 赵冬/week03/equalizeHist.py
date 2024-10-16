"""
@author: 赵冬
@since: 2024年9月26日
"""

import cv2
import numpy as np


# 灰度直方图均衡化
def hist_gray(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dst = equalize(gray)
    cv2.imshow("Histogram Equalization", np.hstack([gray, dst]))
    cv2.waitKey(0)


# 彩色直方图均衡化
def hist_color(img):
    (b, g, r) = cv2.split(img)
    bh = equalize(b)
    gh = equalize(g)
    rh = equalize(r)
    res = cv2.merge((bh, gh, rh))
    cv2.imshow("lenna", np.hstack([img, res]))
    cv2.waitKey(0)


# 直方图均衡化算法
def equalize(channel):
    src_h, src_w = channel.shape
    hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    dst_img = np.zeros((src_h, src_w), dtype=channel.dtype)
    cum_hist = np.cumsum(hist)
    total = src_w * src_h
    for i in range(src_w):
        for j in range(src_h):
            dst_img[j, i] = int(cum_hist[channel[j, i]] / total * 256 - 1)
    return dst_img


if __name__ == '__main__':
    in_img = cv2.imread("lenna.png")
    hist_gray(in_img)
    hist_color(in_img)
