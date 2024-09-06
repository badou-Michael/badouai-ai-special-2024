import numpy as np
import cv2

image = cv2.imread('picture.png')
height, width = image.shape[:2]
print(image)
image_gray = np.zeros((height, width), image.dtype)
for i in range(height):
    for j in range(width):
        m = image[i, j]
        image_gray[i, j] = int(m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)
print(image_gray)
cv2.imshow("gray image", image_gray)

# image = cv2.imread('lenna.png')
# # image_gray = rgb2gray(image)
# image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray image", image_gray)
# print(image_gray)
# cv2.waitKey(0)
cv2.waitKey(0)
