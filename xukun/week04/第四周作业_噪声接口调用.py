import numpy as np
import cv2
from skimage.util import random_noise

img = cv2.imread('lenna.png',0)
cv2.imshow('original',img)
#高斯噪声
gaussian_img=random_noise(img, mode='gaussian')
cv2.imshow('gaussian',gaussian_img)
#椒盐噪声
salt_img = random_noise(img, mode='s&p')
cv2.imshow('salt',salt_img)
cv2.waitKey(0)
