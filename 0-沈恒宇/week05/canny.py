"""
调用接口实现canny算法
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
"""

import cv2
from matplotlib import pyplot as plt


image = cv2.imread('lenna.png')
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny_image_1 = cv2.Canny(gray_image, 200, 300)
canny_image_2 = cv2.Canny(gray_image, 200, 400)
canny_image_3 = cv2.Canny(gray_image, 300, 400)

# cv2.imshow('image', image)
# cv2.imshow('gray_image', gray_image)
# cv2.imshow('canny_image', canny_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(231), plt.imshow(rgb_image), plt.title('Original Image')
plt.subplot(232), plt.imshow(gray_image, cmap='gray'), plt.title('Gray Image')
plt.subplot(234), plt.imshow(canny_image_1, cmap='gray'), plt.title('Canny Image,200-300')
plt.subplot(235), plt.imshow(canny_image_2, cmap='gray'), plt.title('Canny Image,200-400')
plt.subplot(236), plt.imshow(canny_image_3, cmap='gray'), plt.title('Canny Image,300-400')

plt.show()
