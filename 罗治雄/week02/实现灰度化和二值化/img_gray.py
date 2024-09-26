#1.灰度化
import cv2
import numpy as np
img = cv2.imread("E:\work_soft\lenna.png",0)
cv2.imshow('lenna_to_gray',img)
cv2.waitKey()
cv2.destroyAllWindows()



#2.灰度化
import numpy as np
import cv2
img = cv2.imread("E:\work_soft\lenna.png")
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
cv2.imshow("image show gray",img_gray)
cv2.waitKey()
cv2.destroyAllWindows()

#3.灰度化
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
img = cv2.imread("E:\work_soft\lenna.png")
img_gray = rgb2gray(img)
cv2.imshow("image show gray",img_gray)
cv2.waitKey()
cv2.destroyAllWindows()
