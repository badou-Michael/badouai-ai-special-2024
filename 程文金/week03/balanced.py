import cv2
import numpy as np
from matplotlib import pyplot as plt

img_src = cv2.imread("test_img.jpeg")
img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
img_des = cv2.equalizeHist(img_gray)

# hist = cv2.calcHist([img_des],[0],None,[256],[0,256])
#
# plt.figure()
# plt.hist(img_des.ravel(), 256)
# plt.show()

cv2.imshow("Histogram Equalization", np.hstack([img_gray, img_des]))
cv2.waitKey(0)

# (r,g,b) = cv2.split(img_src)
#
# des_r = cv2.equalizeHist(r)
# des_g = cv2.equalizeHist(g)
# des_b = cv2.equalizeHist(b)
#
# img_des = cv2.merge( (des_r,des_g,des_b ) )

# cv2.imshow('gray_test_img', img_des )
# cv2.waitKey(0)
# cv2.destroyAllWindows()