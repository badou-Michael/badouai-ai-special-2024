import cv2 as cv

from skimage import util

img = cv.imread("lenna.png")
img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
noise_gs_img=util.random_noise(img2, mode='s&p', amount=0.8)

cv.imshow("source", img2)
cv.imshow("lenna",noise_gs_img)

cv.waitKey(0)
cv.destroyAllWindows()
