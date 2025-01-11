import cv2
import numpy as np



def function(img):
    height, width, channels = img.shape
    np.zeros

img = cv2.imread("lenna.png")
zoom=function(img)
zoom = cv2.resize(img, (800, 800),interpolation = cv2.INTER_NEAREST)
print(zoom)
print(zoom.shape)
cv2.imshow("nearest interp", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
