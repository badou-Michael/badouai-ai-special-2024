import cv2 as cv
from skimage import util

img = cv.imread("lenna.png")
noise_gs_img = util.random_noise(img, mode='poisson', clip=True)
cv.imshow("source", img)
cv.imshow("lenna_poisson_noise", noise_gs_img)
cv.waitKey(0)
