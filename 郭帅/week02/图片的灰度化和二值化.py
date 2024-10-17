# 1、图片的灰度化

import cv2                                                        #引用opencv
img = cv2.imread("lenna.png")                                     #读取图片
h,w=img.shape[:2]                                                 #读取图片的高和宽，把3通道图片转为单通道图片
# print(h,w)                                                      #输出高和宽
import numpy                                                      #引用numpy,多维数组创建、计算、复杂的操作等
img_gray = numpy.zeros([h,w],img.dtype)                    # numpy.zeros用于创建具有指定形状和类型的新数组的函数
for i in range(h):                                                #for in range 循环的意思
    for j in range(w):
        m = img[i,j]                                              # m 为通过循环取到【h,w】定位坐标的 BGR数值
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)     #通过浮点计算将彩色进行灰度化，注意opencv通道排列为BGR

# print(img_gray)
cv2.imwrite("lenna_gray.png",img_gray)                    # 数组中的数据编码为指定的图像格式,并将图像保存到磁盘的函数调用

# 2、图片的二值化

c,b = img_gray.shape                                              #读取灰度化图片的高和宽
img_two = numpy.zeros([c,b],dtype=numpy.uint8)             #创建一张和灰度化高和宽一致的图片
threshold_value = 128                                             #定义二值化的阈值
for a in range(c):
    for d in range(b):                                            #与阈值比较进行二值化
        if img_gray[a,d] >= threshold_value:
            img_two[a,d] = 255                                    #大于阈值  白
        else:
            img_two[a,d] = 0                                      #小于阈值  黑
cv2.imwrite("lenna_two.png",img_two)                     #根据计算的数组生成图片
