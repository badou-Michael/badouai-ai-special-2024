import cv2 as cv
from skimage import util
img =cv.imread("lenna.png")
img =cv.cvtColor(img,cv.COLOR_BGR2GRAY)
noise_img =util.random_noise(img, mode='gaussian')
cv.imshow("source", img)
cv.imshow("gaussion", noise_img)
cv.waitKey(0)
