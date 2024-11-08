import cv2
import numpy as np

path = "/Users/zhouzhaoyu/Desktop/ai/lenna.png"
img = cv2.imread(path)
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(img, 30,100)
cv2.imshow("canny",canny)
cv2.waitKey(0)
