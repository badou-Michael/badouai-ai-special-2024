
import cv2
import numpy as np


# 最临近插值
def nearestInterp(img,out_dim):
    height,width,channels =img.shape
    dst_height,dst_width = out_dim[0],out_dim[1]
    emptyImage=np.zeros((dst_height,dst_width,channels),np.uint8)
    sh=dst_height/height
    sw=dst_width/width
    for i in range(dst_height):
        for j in range(dst_width):
            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage

img=cv2.imread("lenna.png")

cv2.waitKey(0)


