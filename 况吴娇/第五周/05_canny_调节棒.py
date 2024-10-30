#!/usr/bin/env python
# encoding=gbk

'''
Canny��Ե��⣺�Ż��ĳ���
'''
import cv2
import numpy as np

def CannyThreshold(lowThreshold):
#detected_edges = cv2.GaussianBlur(gray,(3,3),0) #��˹�˲�
#cv2.GaussianBlur(src, ksize, sigmaX, sigmaY=None, borderType=None)
##0 ���Ǹ��� OpenCV �Զ����� X ����� Y ����ı�׼���Ϊ����û��ָ�� Y ����ı�׼����������� X ������ͬ��
 # ��������һ���µ�ͼ�񣬸�ͼ��������ͼ�񾭹���˹ģ�������Ľ����
# sigmaX����˹���� X �����ϵı�׼���� sigmaX Ϊ 0����ô���� ksize ��ȣ�sigmaX �ᱻ�Զ����㡣
#sigmaY������ѡ����˹���� Y �����ϵı�׼����Ϊ None���� sigmaY ������ sigmaX����� sigmaY Ϊ 0����ô���� ksize �߶ȣ�sigmaY �ᱻ�Զ����㡣
    detected_edges = cv2.Canny(gray
                               , lowThreshold
                               , lowThreshold * ratio
                               ,apertureSize=kernel_size)
##ʹ��cv2.Canny�������б�Ե��⡣������������ĸ�������(����ĻҶ�ͼ��,����ֵ������ֵ��Sobel���ӵĿ׾���С����������Ϊ3)
    dst=cv2.bitwise_and(img,img,mask = detected_edges)
    cv2.imshow('canny result',dst)
#mask������ѡ�����������룬��һ��8λ��ͨ��ͼ������ͼ���з������ص�λ�ý��������������ص�λ�ý�������Ϊ�㡣

lowThreshold = 0
max_lowThreshold = 100
ratio = 3
kernel_size = 3

img = cv2.imread('lenna.png')
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('canny result')
#���õ��ڸ�,
'''
�����ǵڶ���������cv2.createTrackbar()
����5����������ʵ������������������ʹ����֪����ʲô��˼��
��һ�������������trackbar���������
�ڶ��������������trackbar����������������
�����������������trackbar��Ĭ��ֵ,Ҳ�ǵ��ڵĶ���
���ĸ������������trackbar�ϵ��ڵķ�Χ(0~count)
������������ǵ���trackbarʱ���õĻص�������
'''
cv2.createTrackbar('Min threshold','canny result',lowThreshold, max_lowThreshold, CannyThreshold)

CannyThreshold(0)  # initialization
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2  ESC ����ASCII ��Ϊ 27��
    cv2.destroyAllWindows()