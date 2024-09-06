import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

def cv_thres(img, thresh=150, max_val=255, type=cv2.THRESH_BINARY):
    ret, dst = cv2.threshold(img, thresh, max_val, type)
    return dst


if __name__ == "__main__":
    # read image
    img = cv2.imread("lenna.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_cvthres = cv_thres(img_gray)

    cv2.imshow("Lenna", img)
    cv2.imshow("CV_THRESH", img_cvthres)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
