import random
import cv2
from skimage import util

#高斯噪声
img=cv2.imread('lena.png')
noise_gauss_image=util.random_noise(img,mode='gaussian')
cv2.imshow('gauss noise',noise_gauss_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#椒盐噪声
img=cv2.imread('lena.png')
noise_gauss_image=util.random_noise(img,mode='s&p')
cv2.imshow('PepperSalt',noise_gauss_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#泊松噪声
img=cv2.imread('lena.png')
noise_gauss_image=util.random_noise(img,mode='poisson')
cv2.imshow('Poisson noise',noise_gauss_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
