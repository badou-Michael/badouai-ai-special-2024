from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

img = cv2.imread('lenna.png')
H,W = img.shape[:2]
img_gray = np.zeros([H,W],dtype=img.dtype)
for r in range(H):
    for c in range(W):
        mark = img[r,c]
        img_gray[r,c] = int(mark[0]*0.11 + mark[1]*0.59 + mark[2]*0.3)
print(img)
print (img_gray)
print("image show gray: %s"%img_gray)
cv2.imshow("image show gray",img_gray)
plt.subplot(221)


plt.subplot(221)
img = plt.imread("lenna.png")
plt.imshow(img)
print("---image lenna----")
print(img)



plt.subplot(222)
plt.imshow(img_gray, cmap='gray')
print("---image gray----")
print(img_gray)
#二值化
img_binary = np.where(img_gray >= 127.5 , 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()
