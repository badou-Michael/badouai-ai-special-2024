#!/usr/bin/env python
# encoding=gbk

import cv2
import numpy as np

'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
��Ҫ������
��һ����������Ҫ�����ԭͼ�񣬸�ͼ�����Ϊ��ͨ���ĻҶ�ͼ��
�ڶ�����������ֵ1��
��������������ֵ2��
'''

img = cv2.imread("../sea.jpg", 1)

# ��ȡ��ʾ���ֱ���
screen_width = 2560
screen_height = 1440


# ����ͼ���С�������ݺ��
aspect_ratio = img.shape[1] / img.shape[0]
if aspect_ratio > 1:  # ����ڸ�
    new_width = screen_width
    new_height = int(screen_width / aspect_ratio)
else:  # �ߴ��ڿ�
    new_height = screen_height
    new_width = int(screen_height * aspect_ratio)

# ����ͼ���С
resized_image = cv2.resize(img, (new_width, new_height))

gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 53, 245)

# �Ҷ�ͼ�ͱ�Ե���ͼˮƽƴ��
combined = cv2.hconcat([gray, edges])
cv2.imshow("canny", combined)
cv2.waitKey()
cv2.destroyAllWindows()
