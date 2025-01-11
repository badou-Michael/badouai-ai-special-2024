import cv2
import numpy as np
from skimage import util


if __name__ =="__main__":
    img = cv2.imread('lenna.png')
    poisson_img = util.random_noise(img,mode='poisson')
    salt_pepper_img = util.random_noise(img,mode='s&p')
    cv2.imshow('source',img)
    cv2.imshow('poisson',poisson_img)
    cv2.imshow('salt_pepper',salt_pepper_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
