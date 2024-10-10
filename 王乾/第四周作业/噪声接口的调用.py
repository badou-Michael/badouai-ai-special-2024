import cv2
import numpy as np
from PIL import Image
from skimage import util

img = cv2.imread("lenna.png")
noise_gs_img = util.random_noise(img,mode="gaussian")
cv2.imshow("窗口1",img)
cv2.imshow("窗口2",noise_gs_img)
cv2.waitKey()
