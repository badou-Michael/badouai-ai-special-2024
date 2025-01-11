import cv2
import numpy as np


img_src = cv2.imread("test_img.jpeg")
img_des = np.zeros([1000,1000,3], np.uint8 )

src_w = img_src.shape[0]
src_h = img_src.shape[1]

scalex = img_des.shape[0]/img_src.shape[0]
scaley = img_des.shape[0]/img_src.shape[0]
for k in range(3):
    for i in range(1000):
        for j in range(1000):
              src_x = (i+0.5) / scalex - 0.5
              src_y = (j+0.5) / scaley - 0.5

              src_x1 = int(src_x)
              src_y1 = int(src_y)
              src_x2 = min( src_x1+1, src_w-1)
              src_y2 = min( src_y1+1, src_h-1)

              temp1 = (src_x2-src_x) * img_src[src_x1,src_y1, k] + (src_x - src_x1) * img_src[src_x2,src_y1, k]
              temp2 = (src_x2-src_x) * img_src[src_x1,src_y2, k] + (src_x - src_x1) * img_src[src_x2,src_y2, k]

              img_des[i,j, k] =  (src_y2 - src_y) * temp1 + (src_y - src_y1) * temp2


cv2.imshow('gray_test_img', img_des )
cv2.waitKey(0)
cv2.destroyAllWindows()