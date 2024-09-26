import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv2.imread("lenna.png")
noise_gs_img = util.random_noise(img, mode='gaussian')
noise_sp_img = util.random_noise(img, mode="s&p")

cv2.imshow("origin", img)
cv2.imshow("gauss", noise_gs_img)
cv2.imshow("saltandpepper", noise_sp_img)
#cv.imwrite('lenna_noise.png',noise_gs_img)
cv2.waitKey(0)
cv2.destroyAllWindows()