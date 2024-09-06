import cv2
from skimage import util

img = cv2.imread("lenna.png")
distImg = util.random_noise(img, mode='salt')
cv2.imshow("img",img)
cv2.imshow("distImg",distImg)
cv2.waitKey(0)