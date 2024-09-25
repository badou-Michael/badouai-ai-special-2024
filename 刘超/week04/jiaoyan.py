import numpy as np
import cv2

percent = 0.8

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

h, w = img.shape[:2]
#img_j = np.zeros([h, w], img.dtype)
img_j = np.array(img)

print(img_j)

num = np.floor(img.size * 0.8)
for _ in range(int(num)):
    x, y = np.random.randint(0, h), np.random.randint(0, w)
    img_j[x][y] = 255
print(img_j)

cv2.imshow('original', img)
cv2.imshow('jiaoyan', img_j)
cv2.waitKey(0)
