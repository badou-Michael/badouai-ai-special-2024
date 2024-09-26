# -*- coding: utf-8 -*-
import cv2
import numpy as np

def resize(img, dstHeight, dstWidth):
    h, w, c = img.shape
    newImg = np.zeros((dstHeight, dstWidth, c),np.uint8)
    sh = dstHeight/h
    sw = dstWidth/w
    for i in range(dstHeight):
        for j in range(dstWidth):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            newImg[i,j]=img[x,y]
    return newImg


img = cv2.imread("../lenna.png")
newImg = resize(img, 800, 800)
cv2.imshow("nearest 800", newImg)
newImg2 = resize(img, 400, 400)
cv2.imshow("nearest 400", newImg2)
cv2.waitKey(0)