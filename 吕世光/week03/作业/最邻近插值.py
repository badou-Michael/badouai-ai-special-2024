import cv2
import numpy as np


def function(srcImg,height,width):
    h, w, c = srcImg.shape
    newImg = np.zeros((height, width, c), srcImg.dtype)
    scaleH = h / height
    scaleW = w / width
    for i in range(height):
        for j in range(width):
            src_x = int(round(j * scaleW))
            src_y = int(round(i * scaleH))
            newImg[i, j] = srcImg[src_y, src_x]
    return newImg


img = cv2.imread("lenna.png")
scaleImg = function(img,800, 800)

cv2.imshow("origin", img)
cv2.imshow("scaleImg", scaleImg)
cv2.waitKey(0)
