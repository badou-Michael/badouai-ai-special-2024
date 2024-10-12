import cv2
import numpy as np

image_src = cv2.imread('test_img.jpeg')
w, h = image_src.shape[:2]
two_value_img = np.zeros([w,h], np.uint8)
for i in range(w):
     for j in range(h):
         m = image_src[i, j]
         gray = (m[0] * 0.11 + m[1] * 0.59 + m[2] * 0.3)/255
         if gray >= 0.5 : two_value_img[i,j] = 255
         else : two_value_img[i,j] = 0

cv2.imshow('gray_test_img', two_value_img )
cv2.waitKey(0)
cv2.destroyAllWindows()