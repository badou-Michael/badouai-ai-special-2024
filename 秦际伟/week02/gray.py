import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from skimage.color import rgb2gray

# 灰度读取
img = cv.imread('lenna.png', 0)
cv.imshow("lenna", img)
cv.waitKey(3000)


# 灰度化
img_p = cv.imread('lenna.png')
img_gray = rgb2gray(img_p)
plt.imshow(img_gray, cmap='gray')
plt.show()

# 二值化
print(img_gray)
img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_binary)
print(img_binary.shape)

