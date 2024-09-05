import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

def flt_alg(img):
    # convert RGB to GRAY
    height, width, channels = img.shape
    gray_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            # get the value of R G B
            r, g, b = img[i, j]
            # calculate the value of GRAY
            gray_img[i][j] = int(r * 0.3 + g * 0.59 + b * 0.11)
    return gray_img

def binary_alg(img):
    # convert GRAY to BINARY
    height, width = img.shape
    binary_img = np.zeros((height, width), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            binary_img[i,j] = (0 if (img[i,j]/255) <= 0.5 else 1)
    return binary_img


# read the image
img = cv.imread('lenna.png')

# convert BGR to RGB
img_RGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# check the image
plt.subplot(2,2,1)
plt.imshow(img_RGB)
plt.axis('off')
# plt.show()

# STEP 1: convert RGB to GRAY
# METHOD 1 - floating algorithm
# gray_img = flt_alg(img_RGB)
# plt.subplot(2,2,2)
# plt.imshow(gray_img, cmap='gray')
# plt.axis('off')
# # plt.show()

# METHOD 2 - COLOR_BGR2GRAY
gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
plt.subplot(2,2,2)
plt.imshow(gray_img, cmap='gray')
plt.axis('off')
# plt.show()


# STEP 2: convert GRAY to BIANRY
binary_img = binary_alg(gray_img)
plt.subplot(2,2,3)
plt.imshow(binary_img, cmap='gray')
plt.axis('off')
plt.show()