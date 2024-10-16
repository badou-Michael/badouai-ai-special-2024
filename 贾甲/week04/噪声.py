#!/usr/bin/env python
#-*- coding:utf-8 -*-
#author JiaJia time:2024-09-23
import cv2 as cv
import numpy as np
from PIL import  Image
from skimage import  util


img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='poisson')

cv.imshow("source",img)
cv.imshow("lenna",noise_gs_img)

cv.waitKey()
cv.destroyAllWindows()


