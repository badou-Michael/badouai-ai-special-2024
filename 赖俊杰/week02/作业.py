from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

image = imread("1.png")
h,w = image[:2]
for i in h:
    for j in w:
        m=image[i,j]
        image_gray=int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3) 
plt.subplot(121)
plt.imshow(image_gray, cmap='gray')

h1,w1 = image_gray.shape
for i in h1:
    for j in w1:
        if image_gray[h1,w1]>=0.5:
            image_gray[h1, w1]=1
        else:
            image_gray[h1, w1]=0
plt.subplot(122)
plt.imshow(image_gray, cmap='gray')
plt.show()
