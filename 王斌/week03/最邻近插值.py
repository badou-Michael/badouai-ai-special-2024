
import cv2
import numpy as np

def function(img):
    height, width, channel = img.shape
    imageInterp = np.zeros((400,400,channel),np.uint8)
    h = height/400
    w = width/400
    for i in range(400):
        for j in range(400):
           x = int(i*h+0.5)
           y = int(j*w+0.5)
           imageInterp[i,j] = img[x,y]
    return imageInterp
img = cv2.imread("C:/Users/Administrator/Desktop/123.jpg")
zoom=function(img)
cv2.imshow("nearest interp",zoom)
cv2.imshow("image",img)
cv2.waitKey(0)
