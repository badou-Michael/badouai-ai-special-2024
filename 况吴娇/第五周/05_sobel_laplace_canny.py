#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread("lenna.png",1)  #1��ʾ�Բ�ɫģʽ��ȡͼ��
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


#Sobel ������һ������ͼ�������ɢ΢�����ӣ�������˸�˹ƽ����΢���󵼣����ڼ���ͼ�����ȵĿռ��ݶȣ��Ӷ����ͼ���еı�Ե��
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
# 1, 0��0, 1���ֱ��ʾ��x��y�����ϵ��󵼽�����
# ksize=3��Sobel���ӵĴ�С������ʹ��3x3�ĺˡ�
img_sobel_x=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)  #cv2.CV_64F��Ŀ��ͼ�����ȣ�ʹ��64λ��������
img_sobel_y=cv2.Sobel(img_gray,cv2.CV_64F,1,0,ksize=3)


# Laplace ����
img_laplace = cv2.Laplacian(img_gray, cv2.CV_64F, ksize=3)  ##laplace��������һ�ֱ�Ե��ⷽ������ͨ������ͼ��Ķ��׵���������Ե��

# Canny ����
img_canny = cv2.Canny(img_gray, 100, 150)  #Canny������һ�ַǳ����еı�Ե����㷨����ͨ����׶δ���������Ե��

#plt.imshow(img_gray, "gray")  # ʹ�ûҶ���ɫӳ����ʾͼ��
plt.subplot(231),plt.imshow(img_gray, "gray"),plt.title('original')
plt.subplot(232),plt.imshow(img_sobel_x, "gray"),plt.title('Sobel_x')
plt.subplot(233),plt.imshow(img_sobel_y, "gray"),plt.title('Sobel_y')
plt.subplot(234),plt.imshow(img_laplace, "gray"),plt.title('img_laplace')
plt.subplot(235),plt.imshow(img_canny, "gray"),plt.title('img_canny')


plt.show()


