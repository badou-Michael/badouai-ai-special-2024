import cv2
from skimage import util

img=cv2.imread("lenna.png")
noise_img=util.random_noise(img,mode="salt")
cv2.imshow("noise_img",noise_img)
cv2.imshow("img",img)
cv2.waitKey(0)
