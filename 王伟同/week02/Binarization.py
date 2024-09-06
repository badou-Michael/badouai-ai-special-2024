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
cv2.waitKey(0)
