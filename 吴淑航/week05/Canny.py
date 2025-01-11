import cv2
import numpy as np

img= cv2.imread("lenna.png")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
result= cv2.Canny(gray,200,300)
cv2.imshow("canny",result)
cv2.waitKey()
