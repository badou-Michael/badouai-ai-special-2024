import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

''' bilinear interpolation '''

def bilinear(img,w2,h2):
    w1,h1,c=img.shape
    w12=w1/w2
    h12=h1/h2
    imgBilinear=np.zeros((w2,h2,c),np.uint8)
    for i in range(w2):
        for j in range(h2):
            for k in range(c):
                x=(i + 0.5)*w12 - 0.5
                y=(j + 0.5)*h12 - 0.5
                xt = int(np.floor(x))
                yt = int(np.floor(y))

                x0 = max(xt, 0) #np.floor(x)是float不是int,,,min()和max()不超出边界范围0~w1-1,0~h1-1
                y0 = max(yt, 0)
                x1 = min(xt + 1, w1 - 1) #不是w2和h2
                y1 = min(yt + 1, h1 - 1)

                Q0 = (x1 - x) * img[x0, y0, k] + (x - x0) * img[x1, y0, k]
                Q1 = (x1 - x) * img[x0, y1, k] + (x - x0) * img[x1, y1, k]
                Q = (y1 - y) * Q0 + (y - y0) * Q1
                imgBilinear[i,j,k]=int(Q) #img是uint8不是float
    return imgBilinear

img = cv2.imread("lenna.png")

imgBilinear=bilinear(img,300,600) #300,300  700,700  300,700  700,300
print(img.shape)
print(imgBilinear.shape)

cv2.imshow("imgBilinear", imgBilinear)
cv2.imshow("img", img)
cv2.waitKey(0)
