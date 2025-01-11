
import numpy as np
import cv2

#读取图片
img=cv2.imread('photo.jpg')
#找到图片中文件的四个角，命名为src
src=np.float32([[127,316],[2543,176],[215,3739],[2727,3699]])
#设置a4文件的四个角，命名为dst
dst=np.float32([[0,0],[595,0],[0,841],[595,841]])

#获取warpmatrix
wm=cv2.getPerspectiveTransform(src,dst)
print('WarpMatrix')
print(wm)
#实现透视变换
result=cv2.warpPerspective(img,wm,(595,841))
#展示结果
cv2.imshow('src',img)
cv2.imshow('result',result)
cv2.waitKey(0)
