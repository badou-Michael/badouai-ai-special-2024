import cv2 as cv
import numpy as np
from PIL import Image
from skimage import util
image = cv.imread("lenna.png")
Gauss_noise_image=util.random_noise(image,mode='poisson')
sp_noise_image=util.random_noise(image,mode='s&p')
imgs=np.hstack([Gauss_noise_image,sp_noise_image])
cv.imshow("Noises", imgs)
#cv.imshow("lenna",image)
cv.waitKey()
