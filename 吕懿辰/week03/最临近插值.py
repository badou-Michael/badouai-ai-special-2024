import cv2
import numpy as np
def function(img):
    height,width,channels =img.shape
    emptyImage=np.zeros((size1,size1,channels),np.uint8)   ##size1为设置的放大的图片大小
    sh=size1/height    ##放大的比例
    sw=size1/width
    for i in range(size1):
        for j in range(size1):
            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整,用于四舍五入。
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage
    
# cv2.resize(img, (size1,size1,c),near/bin)

img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)   ##cv2使用这种方式显示图片


