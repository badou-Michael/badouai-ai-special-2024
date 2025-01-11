import cv2
import numpy as np

img=cv2.imread('photo1.jpg')
result3 = img.copy() ##这行代码创建了 img 的一个副本，用于进行透视变换。这样做是为了避免在原图上直接进行操作。

'''
注意这里src和dst的输入并不是图像，而是图像对应的顶点坐标。
'''

src= np.float32([[207, 151], [517, 285], [17, 601], [343, 731]])
dst= np.float32([[0, 0], [337, 0],[0, 488], [337, 488]])
#OpenCV 的某些函数要求输入数据类型为 np.float32，因为它们需要进行浮点数计算。使用 np.float32 可以确保与 OpenCV 函数的兼容性，并提高计算效率。
print(img.shape) #打印出原始图像的尺寸，即高度、宽度和颜色通道数
# 生成透视变换矩阵；进行透视变换 #\
m=cv2.getPerspectiveTransform(src,dst)
# getPerspectiveTransform 函数根据提供的源点和目标点计算透视变换矩阵。透视变换是一种几何变换，它可以模拟相机视角的变化，
# 使得图像中的某些部分看起来像是从不同的视角或距离观察到的。

#打印透视变换矩阵
print("warpMatrix:",m)
#应用透视变换

result=cv2.warpPerspective(result3,m,[337,488])
#这行代码使用 OpenCV 的 warpPerspective 函数将透视变换矩阵 m 应用到图像 result3 上，生成变换后的图像 result。
# 目标图像的尺寸被设置为 (337, 488)，这应该是 dst 坐标定义的矩形区域的尺寸。
#warpPerspective 函数是 OpenCV 提供的一种应用透视变换的方法。它根据提供的变换矩阵和目标图像尺寸，对源图像进行透视变换。
cv2.imshow('src',img)
cv2.imshow('result_透视变换',result)
cv2.waitKey(0)
