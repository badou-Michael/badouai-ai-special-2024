import cv2
import numpy as np


def function(img):
    height,width,channels =img.shape
    emptyImage=np.ones((800,800,channels),np.uint8)*0
    #print(emptyImage.shape[2]

    sh=emptyImage.shape[0]/height
    sw=emptyImage.shape[1]/width
    for i in range(emptyImage.shape[0]):
        for j in range(emptyImage.shape[1]):
            x=int(i/sh + 0.5)  #int(),转为整型，使用向下取整。
            y=int(j/sw + 0.5)
            emptyImage[i,j]=img[x,y]
    return emptyImage
    
# cv2.resize(img, (800,800,c),near/bin)

img=cv2.imread("C:/Users/DMR/Desktop/1.png")
zoom=function(img)
#print(zoom)
#print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)


