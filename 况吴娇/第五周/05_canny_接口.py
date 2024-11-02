#! /user/bin/envs python
#coding=gbk


###方法1，调接口
import cv2

img=cv2.imread('lenna.png',1) #这行代码使用cv2.imread函数读取名为"lenna.png"的图像文件。参数1表示以彩色模式读取图像，如果参数是0，则以灰度模式读取。
gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #这行代码将读取的彩色图像转换为灰度图像。cv2.cvtColor函数用于转换图像的颜色空间，这里从BGR（OpenCV默认的颜色顺序）转换为灰度。
cv2.imshow('canny',cv2.Canny(gray_img,200,300))
#200 是 threshold1，即较低的阈值。300 是 threshold2，即较高的阈值。
# cv2.waitKey(0)#无期限等待
# cv2.destroyAllWindows()
'''
cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient ]]])   
必要参数：
第一个参数是需要处理的原图像，该图像必须为单通道的灰度图；
第二个参数是阈值1；
第三个参数是阈值2。
'''


cv2.waitKey(0)#无期限等待
