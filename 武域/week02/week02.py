import numpy as np
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# grey scale
image = cv2.imread("lenna.png")
# get the shape of image: hight, width, layer
h,w,l = image.shape
image_gray = np.zeros([h,w],image.dtype)

for i in range(h):
    for j in range(w):
        val = image[i,j]
        image_gray[i,j] = int(val[0] * 0.11 + val[1] * 0.59 + val[2] * 0.3)

cv2.imwrite('gray_lenna.png', image_gray)

image_bin = np.where(image_gray >= 255/2, 255, 0)
cv2.imwrite('bin_lenna.png', image_bin)


