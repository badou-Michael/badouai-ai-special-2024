import cv2
import numpy as np

img = cv2.imread("test_img.jpeg", 1)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cannyimg = cv2.Canny(gray, 50, 300)
cv2.imshow("canny", cannyimg)
cv2.waitKey()
cv2.destroyAllWindows()