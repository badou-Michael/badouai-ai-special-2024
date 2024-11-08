import cv2
import numpy as np

img = cv2.imread('lenna.png')
cv2.imshow('Original', img)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

canny_img = cv2.Canny(gray_img, 25, 130);

cv2.imshow('Canny', canny_img)
cv2.waitKey(0)
