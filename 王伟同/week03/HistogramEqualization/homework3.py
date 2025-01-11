import numpy as np
import cv2

image = cv2.imread('lenna.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
equalized = cv2.equalizeHist(gray)
# blue, green, red = cv2.split(image)
# blue = cv2.equalizeHist(blue)
# green = cv2.equalizeHist(green)
# red = cv2.equalizeHist(red)
# finalPicture = cv2.merge([blue, green, red])
# cv2.imshow('final', finalPicture)
cv2.imshow('gray', gray)
cv2.imshow('equalized', equalized)
cv2.waitKey(0)
