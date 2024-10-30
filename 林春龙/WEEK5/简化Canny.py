import cv2
import numpy as np 


lowThreshold = 0  
max_lowThreshold = 100  
ratio = 3  
kernel_size = 3  
  
img = cv2.imread('lenna.png')  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #转换彩色图像为灰度图
  
cv2.namedWindow('canny result')  

#调节trackbar时调用的回调函数
def CannyThreshold(lowThreshold):  
    #detected_edges = cv2.GaussianBlur(gray,(3,3),0) #高斯滤波 
    detected_edges = cv2.Canny(gray,
            lowThreshold,
            lowThreshold*ratio,
            apertureSize = kernel_size)  #边缘检测
    
    dst = cv2.bitwise_and(img,img,mask = detected_edges)  
    cv2.imshow('canny result',dst)

#设置调节杠,
'''
下面是第二个函数，cv2.createTrackbar()
共有5个参数，其实这五个参数看变量名就大概能知道是什么意思了
第一个参数，是这个trackbar对象的名字
第二个参数，是这个trackbar对象所在面板的名字
第三个参数，是这个trackbar的默认值,也是调节的对象
第四个参数，是这个trackbar上调节的范围(0~count)
第五个参数，是调节trackbar时调用的回调函数名
'''
cv2.createTrackbar('Min threshold','canny result',lowThreshold, max_lowThreshold, CannyThreshold)  
  
CannyThreshold(lowThreshold)  # initialization  
if cv2.waitKey(0) == 27:  #wait for ESC key to exit cv2
    cv2.destroyAllWindows()  
