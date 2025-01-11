import numpy as np
import cv2 as cv
import pandas as pd

'''canny边缘检测算法 的接口'''

gray = cv.imread("lenna.png", 0)
gray_canny = cv.Canny(gray, 80, 160)  # 200,300

cv.imshow("gray", gray)
cv.imshow("gray_canny", gray_canny)
cv.waitKey(0)
cv.destroyAllWindows()
