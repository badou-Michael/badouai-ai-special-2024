import cv2
import numpy as np
# python imaging library
from PIL import Image
from skimage import util

src = cv2.imread("../lenna.png")
noise_gauss = util.random_noise(src, mode = 'gaussian')
noise_sp = util.random_noise(src, mode = 's&p')
noise_poisson = util.random_noise(src, mode = 'poisson')

cv2.imshow("source", src)
cv2.imshow("gauss", noise_gauss)
cv2.imshow("salt_pepper", noise_sp)
cv2.imshow("poisson", noise_poisson)

cv2.waitKey(0)
cv2.destroyAllWindows()