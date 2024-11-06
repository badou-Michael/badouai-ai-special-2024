from skimage.color import rgb2gray
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np


img = cv.imread('../week02/lenna.png')
w,h,ch = img.shape
w_img = np.zeros((800,800,ch),np.uint8)
#将原图放大
sh = 800/h
sw = 800/w
for i in range(800):
    for j in range(800):
        x = int(i /sh + 0.5)
        y = int(j/ sw + 0.5)
        w_img[i][j] = img[x][y]


cv.imshow("imag_source",img)
cv.imshow("imag_dest",w_img)
cv.waitKey(0)