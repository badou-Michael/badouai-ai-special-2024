import cv2
import numpy as np
img = cv2.imread('../../../request/photo01.jpg')
img_copy = img.copy()
# 透视坐标的顶点，可以在图片中量出来
src = np.float32([[207, 152],[518, 286],[17, 604],[343, 732]])
dst = np.float32([[0, 0],[338, 0],[0, 488],[338, 488]])
print(img.shape)
'''
cv2.getPerspectiveTransform 是 OpenCV 库中的一个函数，用于计算从一组点到另一组点的透视变换矩阵。
这个函数通常用于图像处理中，当你需要将图像中的一组点映射到另一组点时，比如在图像拼接、图像校正或者图像变换中。
'''
m = cv2.getPerspectiveTransform(src,dst)
print('warpMatrix：\n',m)
'''
cv2.warpPerspective 是 OpenCV 库中的一个函数，用于对图像进行透视变换。这种变换通常用于校正图像的透视失真，
或者将图像从一个平面映射到另一个平面上。这个函数接受一个图像和一个变换矩阵，然后输出变换后的图像。
'''
result = cv2.warpPerspective(img_copy,m,(338,488))
cv2.imshow('source',img)
cv2.imshow('result',result)
cv2.waitKey(0)
