#!/usr/bin/env python
# encoding=gbk

import cv2  
import numpy as np  
from matplotlib import pyplot as plt  

img = cv2.imread("../lenna.png",1)

img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  

'''
Sobel����
Sobel���Ӻ���ԭ�����£�
dst = cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]]) 
ǰ�ĸ��Ǳ���Ĳ�����
��һ����������Ҫ�����ͼ��
�ڶ���������ͼ�����ȣ�-1��ʾ���õ�����ԭͼ����ͬ����ȡ�Ŀ��ͼ�����ȱ�����ڵ���ԭͼ�����ȣ�
dx��dy��ʾ�����󵼵Ľ�����0��ʾ���������û���󵼣�һ��Ϊ0��1��2��
����ǿ�ѡ�Ĳ�����
dst��Ŀ��ͼ��
ksize��Sobel���ӵĴ�С������Ϊ1��3��5��7��
scale�����ŵ����ı���������Ĭ�������û������ϵ����
delta��һ����ѡ������������ӵ����յ�dst�У�ͬ����Ĭ�������û�ж����ֵ�ӵ�dst�У�
borderType���ж�ͼ��߽��ģʽ���������Ĭ��ֵΪcv2.BORDER_DEFAULT��
'''

img_sobel_x = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)  # ��x��
img_sobel_y = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)  # ��y��

# Laplace ����  
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  

# Canny ����  
img_canny = cv2.Canny(img_gray, 100 , 150)  

plt.subplot(231), plt.imshow(img_gray, "gray"), plt.title("Original")  
plt.subplot(232), plt.imshow(img_sobel_x, "gray"), plt.title("Sobel_x")  
plt.subplot(233), plt.imshow(img_sobel_y, "gray"), plt.title("Sobel_y")  
plt.subplot(234), plt.imshow(img_laplace,  "gray"), plt.title("Laplace")  
plt.subplot(235), plt.imshow(img_canny, "gray"), plt.title("Canny")  
plt.show()  
