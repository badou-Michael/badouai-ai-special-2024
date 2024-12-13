#双线性插值算法

import numpy as np
import cv2
img=cv2.imread("lenna.png")
if img is None:
    raise ValueError("无法读取图像文件，请检查文件路径和文件名")
h,w,c=img.shape
dest_img=np.zeros((800,800,c),np.uint8)
sh,sw,sc=dest_img.shape
ah=h/sh
aw=w/sw
for k in range(sc):
    for i in range(800):
        for j in range(800):
            src_x = (j + 0.5) * aw - 0.5      #对齐中心点
            src_y = (i + 0.5) * ah - 0.5

            src_x0 = int(np.floor(src_x))
            src_x1 = min(src_x0 + 1 , w-1)
            src_y0 = int(np.floor(src_y))
            src_y1 = min(src_y0 + 1 , h-1)

            temp_x = (src_x1 - src_x) * img[src_y0,src_x0,k] + (src_x - src_x0) * img[src_y0,src_x1,k]
            temp_y = (src_x1 - src_x) * img[src_y1,src_x0,k] + (src_x - src_x0) * img[src_y1,src_x1,k]
            dest_img[i,j,k] = (src_y1 - src_y) * temp_x + (src_y - src_y0) * temp_y

cv2.imwrite("lenna03-2.png",dest_img)
