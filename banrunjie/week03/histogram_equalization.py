import cv2
import numpy as np

if __name__ =="__main__":
    img = cv2.imread('lenna.png')
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    dst_gray = cv2.equalizeHist(img_gray)
    b,g,r = cv2.split(img)
    b_hist = cv2.equalizeHist(b)
    g_hist = cv2.equalizeHist(g)
    r_hist = cv2.equalizeHist(r)
    dst_color = cv2.merge((b_hist,g_hist,r_hist))
    cv2.imshow('dst gray',dst_gray)
    cv2.imshow('dst color',dst_color)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


