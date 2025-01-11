import cv2
from skimage import util 

path = "/Users/zhouzhaoyu/Desktop/ai/lenna.png"
img = cv2.imread(path)
gauss_img = util.random_noise(img, 'gaussian')
pepper_img = util.random_noise(img, 's&p', amount = 0.8)
cv2.imshow('original image', img)
cv2.imshow('Gaussian noise image', gauss_img)
cv2.imshow('pepper salt noise image',pepper_img)
cv2.waitKey(0)