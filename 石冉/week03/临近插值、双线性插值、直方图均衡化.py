#最临近插值
import cv2
image=cv2.imread('lena.png')
resized_image=cv2.resize(image,(800,800),interpolation=cv2.INTER_NEAREST)
cv2.imshow('Resized_Image',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

#双线性插值
resized_image=cv2.resize(image,(800,800),interpolation=cv2.INTER_LINEAR)
cv2.imshow('Resized_Image',resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


#直方图均衡化
import numpy as np
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) # 灰度化
equ=cv2.equalizeHist(gray_image) #直方图均衡化
des=np.hstack((gray_image,equ)) #水平堆叠两张图片对比
cv2.imshow('Original Picture vs Equalized Picture', des)
cv2.waitKey(0)
cv2.destroyAllWindows()
