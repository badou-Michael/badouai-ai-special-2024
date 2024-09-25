from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
import cv2



image=cv2.imread("lenna.png")
cv2.imshow("image",image)

H,W=image.shape[:2]

image_gray = np.zeros([H,W],image.dtype)
for i in range(H):
    for j in range(W):
        m=image[i,j]
        image_gray[i,j]=int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
cv2.imshow("image_gray",image_gray)

image_gray2=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("image_gray2",image_gray2)
# print(image_gray2)

image_bin=np.where(image_gray2>=0.5*255,255,0)
image_bin=np.array(image_bin,dtype=np.uint8)
cv2.imshow("image_bin",image_bin)
# 上一行需要加上dtype=np.uint8否则会报错
# cv2.error: OpenCV(4.10.0) D:/a/opencv-python/opencv-python/opencv/modules/highgui/src/precomp.hpp:156: error: (-215:Assertion failed) src_depth != CV_16F && src_depth != CV_32S in function 'convertToShow'



img = plt.imread("lenna.png")
plt.subplot(131)
plt.title("lenna")
plt.imshow(img)

image_gray=rgb2gray(img)
plt.subplot(132)
plt.title("image_gray",)
plt.imshow(image_gray,cmap="gray")

image_bin=np.where(image_gray>=0.5,1,0)
plt.subplot(133)
plt.title("image_bin",)
plt.imshow(image_bin,cmap="gray")

plt.show()



cv2.waitKey(10000)
# cv2.destroyWindow()
