#!/usr/bin/env python
# encoding=gbk

import cv2


def cannyThreshold(lowThreshold):
    detected_edges = cv2.Canny(gray,
                               lowThreshold * 2,
                               lowThreshold * 3)
    cv2.imshow('canny', detected_edges)


lowThreshold = 0
max_lowThreshold = 100

img = cv2.imread('lenna.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('canny')
cv2.createTrackbar('Min threshold', 'canny', lowThreshold, max_lowThreshold, cannyThreshold)

cannyThreshold(100)
if cv2.waitKey(0) == 27:
    cv2.destroyAllWindows()
