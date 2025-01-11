import cv2
import numpy as np
from skimage.color import rgb2gray
def function(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width= img_gray.shape
    emptyImage=np.zeros((700,700),np.uint8)
    sh=700/height
    sw=700/width
    for i in range(700):
        for j in range(700):
            x = int(i/sh + 0.5)
            y = int(j/sw + 0.5)
            emptyImage[i,j]=img_gray[x,y]
    return emptyImage

img = cv2.imread("lenna.png")
lenna = function(img)
print(lenna)
print("original image:",img.shape)
print("nearest interp image shape:",lenna.shape)
cv2.imshow("nearest interp image",lenna)
cv2.imshow("original image",img)
cv2.waitKey()
