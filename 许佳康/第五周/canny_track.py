#!/usr/bin/env python
# encoding=gbk
 
'''
Canny��Ե��⣺�Ż��ĳ���
'''
import cv2
import numpy as np 
 
def CannyThreshold(lowThreshold):  
    #detected_edges = cv2.GaussianBlur(gray,(3,3),0) #��˹�˲� 
    detected_edges = cv2.Canny(gray,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #��Ե���
 
     #��ԭʼ��ɫ��ӵ����ı�Ե�ϡ� 
     #��λ���롱����������ÿ������,����������ͼ����Ӧλ�õ�����ֵ�ֱ���а�λ���롱����,����Ľ��ͼ��Ķ�Ӧ����ֵ��Ϊ����������ͼ���Ӧ����ֵ�İ�λ������
     #src1��src2��ʾҪ���а�λ���롱��������������ͼ��
     #mask �ǿ�ѡ���������ָ������Ĥ����ֻ����Ĥ��Ӧλ�õ����ؽ��а�λ���롱�����������ķ���ֵ��ʾ��λ���롱����Ľ����
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  
    cv2.imshow('canny result',dst)  
  
 
lowThreshold = 0  
max_lowThreshold = 100  
ratio = 3  
kernel_size = 3  
  
img = cv2.imread('../lenna.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)  #ת����ɫͼ��Ϊ�Ҷ�ͼ
  
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
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()  
