import cv2
import numpy as np
def upsampled(n,image): #n为图像缩放比例
    height,width,channels=image.shape
    H=height*n
    W=width*n
    emptyImage=np.zeros((H,W,channels),np.uint8)
    for i in range(H):
        for j in range(W):
            x=int(i/n+0.5) #四舍五入
            y=int(j/n+0.5)
            x = min(max(x, 0), height - 1) #边界检查
            y = min(max(y, 0), width - 1)
            emptyImage[i,j]=image[x,y]
    return emptyImage

image=cv2.imread('lenna.png')
print(image.shape)
zoomfactor=3 #设置缩放比例
zoom=upsampled(zoomfactor,image)
cv2.imshow('zoom',zoom)
#cv2.imshow('image',image)
print(zoom.shape)
cv2.waitKey()
