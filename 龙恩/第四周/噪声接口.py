import numpy as np
import cv2
from PIL import Image
from skimage import util

img=cv2.imread('lenna.png')
gaussian_img=util.random_noise(img,mode='gaussian',mean=0,var=0.01)

cv2.imshow("source", img)
cv2.imshow("gaussian",gaussian_img)


ps_img=util.random_noise(img,mode='s&p',amount=0.1, salt_vs_pepper=0.7)
cv2.imshow("p&s",ps_img)
