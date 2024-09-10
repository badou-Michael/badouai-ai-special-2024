import cv2
import numpy as np
#实现最邻近插值算法

def neareat(img):
    img = cv2.imread("lenna.png")
    h,w,c=img.shape
    newimg = np.zeros((800,800,c), np.uint8)
    sh=800/h
    sw=800/w     # 缩放比例
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)     # x为原图的坐标，通过将i(新图坐标)/sh--相当于将新图缩小回去
            y=int(j/sw+0.5)     # 将新图坐标映射回原图坐标去确定此时新图坐标的像素值
            newimg[i,j]=img[x,y]
    return newimg

img=cv2.imread("lenna.png")
zoomin=neareat(img)
cv2.imshow("exist",img)
cv2.imshow("Nearest Neighbor Interpolation-zoomin",zoomin)
cv2.waitKey(0)
# 调接口  zoomin=cv2.resize(img,(800,800),interpolation=cv2.INTER_NEAREST)


