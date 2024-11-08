# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import cv2

#法一
def function(img):
    height,weight,channel=image.shape
    image_new=np.zeros((800,800,channel),img.dtype)#uint8 无符号整数
    sh=800/height
    sw=800/weight          #对原始的宽高放大了多少
    for i in range(800):
        for j in range(800):
            x=int(i/sh+0.5)#x坐标值一定是个整数，因为int()是个向下取整的过程。所以要+0.5。为了模拟四舍五入
            y=int(j/sw+0.5)#换算成了在原图里的坐标,缩小为原图
            image_new[i,j]=image[x,y]#直接等于原图的最邻近像素值
    return image_new       #定义函数不能忘记return!

image=cv2.imread('lenna.png')
zoom=function(image)
cv2.imshow("show zoom",zoom)
cv2.waitKey(0)

#法2
resized_image=cv2.resize(image,(800,800),interpolation=cv2.INTER_NEAREST)
cv2.imshow("show resized_image",resized_image)
cv2.waitKey(0)
