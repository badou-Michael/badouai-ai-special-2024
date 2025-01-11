# -*- coding: utf-8 -*-

import cv2
import numpy as np

#最临近插值
# img = cv2.imread("lenna.png")
# img_near = cv2.resize(img,(900,900),interpolation=cv2.INTER_NEAREST)
# cv2.imshow("nearest",img_near)
# cv2.waitKey(0)

def function(img,out_dim):
    src_h, src_w, channel = img.shape
    dst_h, dst_w = out_dim[1], out_dim[0]
    dst_img = np.zeros((dst_h,dst_w,3),dtype=np.uint8)
    sh = dst_h / src_h
    sw = dst_w / src_w
    for dst_x in range(dst_h):
        for dst_y in range(dst_w):
            src_x = int(dst_x / sh + 0.5)
            src_y = int(dst_y / sw + 0.5)
            dst_img[dst_x, dst_y] = img[src_x, src_y]
    return dst_img
img=cv2.imread("lenna.png")
dst = function(img,(800,800))
cv2.imshow("nearest",dst)
cv2.imshow("image",img)
cv2.waitKey(0)
