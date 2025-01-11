import cv2
import numpy as np


img_src = cv2.imread("test_img.jpeg")
img_des = np.zeros([1000,1000,3], np.uint8 )

scalex = img_des.shape[0]/img_src.shape[0]
scaley = img_des.shape[0]/img_src.shape[0]

for i in range(1000):
    for j in range(1000):
          src_x = int(i / scalex)
          src_y = int(j / scaley)
          img_des[i,j] = img_src[src_x,src_y]

cv2.imshow('gray_test_img', img_des )
cv2.waitKey(0)
cv2.destroyAllWindows()
