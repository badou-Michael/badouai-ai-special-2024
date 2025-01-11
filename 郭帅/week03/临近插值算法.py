#临近插值算法

import numpy as np
import cv2
img=cv2.imread("lenna.png")
h,w,c=img.shape
img_nearest=np.zeros((800,800,c),np.uint8)
sw=800/w
sh=800/h
for i in range(800):
    for j in range(800):
        x=int(i/sh + 0.5)
        y=int(j/sw + 0.5)
        img_nearest[i,j]=img[x,y]

cv2.imwrite("lenna03_1.png",img_nearest)
