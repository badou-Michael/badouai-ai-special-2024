import cv2

img=cv2.imread("q.jpg"); #step1 读取图片
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #将彩色图片灰度化

_,binaryImg=cv2.threshold(img,127,255,cv2.THRESH_BINARY)#threshold函数返回两个值，忽略第一个，只接收二值化处理后的图像

cv2.imshow('Gray image',img) #显示灰度图像
cv2.imshow('binary image',binaryImg) #显示二值化图像
cv2.waitKey(0); #等待用户按键，保持窗口打开
cv2.destroyAllWindows() #关闭所有openCV创建的窗口
