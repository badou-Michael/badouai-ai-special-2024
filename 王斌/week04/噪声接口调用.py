import cv2
from skimage import util

img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
noise=util.random_noise(img, mode='salt')

cv2.imshow("source", img)
cv2.imshow("noise",noise)
cv2.waitKey(0)
cv2.destroyAllWindows()
