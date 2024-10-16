
import cv2 as cv
from skimage import util

img = cv.imread("test_img.jpeg")
noice_img = util.random_noise(img, mode='poisson')

cv.imshow("source", img)
cv.imshow("noice_img", noice_img)

cv.waitKey(0)