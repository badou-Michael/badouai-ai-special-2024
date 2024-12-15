#透视变换

import cv2
import numpy as np

#读取图片
img = cv2.imread('1.png')
#源图片像素点
src = np.float32([[38,66],[462,66],[38,488],[462,488]])
#目标图片像素点
det = np.float32([[0,0],[424,0],[0,422],[424,422]])
#计算变换矩阵
m = cv2.getPerspectiveTransform(src,det)
#将矩阵应用到图像上，得到新图
T_img = cv2.warpPerspective(img,m,[424,422])
#保存新图
cv2.imwrite('week06_1.png',T_img)
