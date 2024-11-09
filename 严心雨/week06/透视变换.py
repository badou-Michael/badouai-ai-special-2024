import cv2
import numpy as np

image=cv2.imread('photo1.jpg')

'''
注意这里scc和dst的输入并不是图像，而是图像对应的顶点坐标。
'''
scr=np.float32([[216,159],[527,296],[18,613],[346,740]])
dst=np.float32([[0,0],[337,0],[0,448],[337,448]])

#获取透视变换矩阵函数
#cv2.getPerspectiveTransform 该函数要求输入和输出的点集都应具有相同的形状并为float32
m=cv2.getPerspectiveTransform(scr,dst)
#应用透视变换
result=cv2.warpPerspective(image,m,(337,448))
cv2.imshow("scr：",image)
cv2.imshow("dst：",result)
cv2.waitKey(0)

