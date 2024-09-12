import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread("lenna.png")
#img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
H,W = img.shape[:2]
print(H,W)
M = 800
img_out = np.ones((M,M,3),img.dtype)
#最邻近插值
scal_h = M/H
scal_w = M/W
for i in range(M):
    for j in range(M):
        x = min(int(i / scal_h + 0.5),H-1)
        y = min(int(j / scal_h + 0.5),W-1)
        img_out[i][j] = img[x][y]
#双线性插值
img_out2 = np.ones((M,M,3),img.dtype)

for i in range(M):
    for j in range(M):
        x = i / scal_h
        y = j / scal_h
        x0 = min(int(i / scal_h), H - 1)
        y0 = min(int(j / scal_h), W - 1)
        x1 = min(x0 + 1, H - 1)
        y1 = min(y0 + 1, W - 1)
        temp1 = (x1 - x) * img[x0][y1] + (x - x0) * img[x1][y1]
        temp2 = (x1 - x) * img[x0][y0] + (x - x0) * img[x1][y0]
        img_out2[i][j] = (y - y0) * temp1 + (y1 - y) * temp2

cv.imshow('img',img)
cv.imshow('img2',img_out)
cv.imshow('img3',img_out2)
cv.waitKey(0)
