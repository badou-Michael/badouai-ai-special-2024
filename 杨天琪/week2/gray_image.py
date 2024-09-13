import cv2
import numpy as np

img_org = cv2.imread("lenna.png")
print(img_org)
h, w = img_org.shape[:2]
img_gray = np.zeros([h, w], img_org.dtype)
for i in range(h):
    for j in range(w):
        m = img_org[j, i]
        img_gray[j, i] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

cv2.imshow("image show gray", img_gray)

img_binary = np.zeros([h, w], img_gray.dtype)
for i in range(h):
    for j in range(w):
        if img_gray[j, i] >= int(255/2):
            img_binary[j, i] = 255
        else:
            img_binary[j, i] = 0
cv2.imshow("image show binary", img_binary)


cv2.waitKey()
