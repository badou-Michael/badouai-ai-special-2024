'''
@Project ：BadouCV 
@File    ：noise_APITest.py
@IDE     ：PyCharm 
@Author  ：zjd
@Date    ：2024/10/15 00:02 
'''
import cv2 as cv

from skimage import util


img = cv.imread("../lenna.png")
noise_gs_img=util.random_noise(img,mode='poisson')

cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)

cv.waitKey(0)
cv.destroyAllWindows()
