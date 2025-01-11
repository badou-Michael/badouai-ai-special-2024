import cv2 as cv
import numpy as np

img = imread("lenna.png")

H, W = img.shape[:2]
img_out = np.ones([H,W],img.dtype) 

#灰度化
for i in range(H):
  for j in range(W):
    m = img[i,j]
    img_out[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)

#二值化
img_out2 = np.ones([H,W],img.dtype) 

for i in range(H):
  for j in range(W):
    m = img_out1[i][j]
    if (m <= 123):
      img_out2 = 0
    else:
      img_out2 = 255

plt.figure()

