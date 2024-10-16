import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util

img = cv.imread("lenna.png")
noise_gs_img=util.random_noise(img,mode='gaussian')

cv.imshow("source", img)
cv.imshow("lenna",noise_gs_img)
#cv.imwrite('lenna_noise.png',noise_gs_img)
cv.waitKey(0)
cv.destroyAllWindows()
