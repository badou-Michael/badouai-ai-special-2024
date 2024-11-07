import cv2
import numpy as np

img = cv2.imread('photo1.jpg')
img1 = img.copy()

#原图像素坐标
src = np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
#变换后的像素坐标
dst = np.float32([[0, 0], [337, 0], [0, 488], [337, 488]])

#生成透视变换矩阵
m = cv2.getPerspectiveTransform(src,dst)

#进行透视变换
res = cv2.warpPerspective(img1,m,(337,488))

cv2.imshow('img',img)
cv2.imshow('res',res)
cv2.waitKey(0)
