# -*- coding: utf-8 -*-
# time: 2024/10/26 16:04
# file: Canny.py
# author: flame
import cv2
import numpy as np


if __name__ == '__main__':
    img = cv2.imread("lenna.png")
    # openCV默认是BGR颜色
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imshow("img_gray",img_gray)
    cv2.imshow("canny",cv2.Canny(img_gray,100,150))
    cv2.waitKey()
    cv2.destroyAllWindows()