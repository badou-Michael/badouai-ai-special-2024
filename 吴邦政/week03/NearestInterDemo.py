import cv2
import numpy as np



image = cv2.imread("image.jpg")
h,w,c = image.shape
ch = 800
cw = 800
changeImage = np.zeros((ch,cw,c),np.uint8)
sh = ch/h
sw = cw/w

for i in range(ch):
    for j in range(cw):
        x = int(i/sh + 0.5)
        y = int(j/sw + 0.5)
        changeImage[i,j] = image[x,y]

cv2.imshow("1",changeImage)
cv2.imshow("2",image)
cv2.waitKey(0)

