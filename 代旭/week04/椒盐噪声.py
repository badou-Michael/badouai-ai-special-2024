import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv2.imread("lenna.png")
noise_sp = util.random_noise(img,mode="s&p")
noise_gs = util.random_noise(img,mode="poisson")
cv2.imshow("source",img)
cv2.imshow("sp_lenna",noise_sp)
cv2.imshow("gs_lenna",noise_gs)
cv2.waitKey(0)
