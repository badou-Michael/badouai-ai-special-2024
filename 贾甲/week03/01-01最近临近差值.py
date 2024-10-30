import cv2
import numpy as np
def function(img):
    
    height,width,chananls = img.shape
    emptyImage=np.zeros((800,800,chananls),np.unit8)
    sh=800/height
    sw=800/width
    for i in range(800)
        for j in rang(800):
            x=int(i/sh+0.5)
            y=int(j/sw+0.5)
            emptyImage[i,j]=img[i,j]
    return emptyImage

#cv2.resize(img,(800,800,c),near/bin)

img=cv2.imread("lenna.png")
zoom=function(img)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)

