import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv.imread('lenna.png')
noise_gs_image = util.random_noise(img,mode='poisson')

cv.imshow("source",img)
cv.imshow("lenna",noise_gs_image)
cv.waitKey(0)
cv.destroyAllWindows()
