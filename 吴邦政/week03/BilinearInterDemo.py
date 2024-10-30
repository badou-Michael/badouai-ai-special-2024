import cv2
import numpy as np


image = cv2.imread("image.jpg")
h,w,c = image.shape
ch = 800
cw = 800
changeImage = np.zeros((ch,cw,3),dtype=np.uint8)
sh = float(h)/ch
sw = float(w)/cw
for i in range(c):
    for ch1 in range(ch):
        for cw1 in range(cw):
            x = (cw1 + 0.5) * sw - 0.5
            y = (ch1 + 0.5) * sh - 0.5
            x0 = int(np.floor(x))
            x1 = min(x0 + 1,w -1)
            y0 = int(np.floor(y))
            y1 = min(y0 + 1,h - 1)
            t0 = (x1 - x) * image[y0,x0,i] + (x - x0) * image[y0,x1,i]
            t1 = (x1 - x) * image[y1,x0,i] + (x - x0) * image[y1,x1,i]
            changeImage[ch1,cw1,i] = int((y1 - y) * t0 + (y - y0) * t1)

cv2.imshow("1",image)
cv2.imshow("2",changeImage)
cv2.waitKey(0)