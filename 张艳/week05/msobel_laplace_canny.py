import numpy as np
import cv2
import pandas as pd
from matplotlib import pyplot as plt

'''三种边缘检测算法 的接口'''

gray = cv2.imread("lenna.png", 0)

gray_sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gray_sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
gray_laplace = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
gray_canny = cv2.Canny(gray, 80, 160)  # 200,300

plt.subplot(231), plt.imshow(gray), plt.title('gray')
plt.subplot(232), plt.imshow(gray_sobel_x), plt.title('gray_sobel_x')
plt.subplot(233), plt.imshow(gray_sobel_y), plt.title('gray_sobel_y')
plt.subplot(234), plt.imshow(gray_laplace), plt.title('gray_laplace')
plt.subplot(235), plt.imshow(gray_canny), plt.title('gray_canny')
plt.show()
