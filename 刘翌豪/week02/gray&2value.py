import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def bgr2gray(m):
    gray_value = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
    return gray_value
'''
def two_val(m):
    value2 = bgr2gray(m)/255
    return value2
'''
#读取图像
img = cv2.imread("lenna.png")
#获取图像矩阵大小
height, width = img.shape[0:2]
#建立一个新的单通道记录每个点的灰度值，数据类型与img相同
img2 = np.zeros([height,width],img.dtype) #np.zeros([x,y]'矩阵大小',dtype'新创建的数组数据类型')
#根据公式将每个点的rgb转换为灰度,opencv为BGR格式
for i in range(height):
    for j in range(width):
        rgb_point = img[i,j]
        img2[i,j] = bgr2gray(rgb_point) #转灰
        #img2[i,j] = two_val(rgb_point)#二值化
print(rgb_point)
#np.where(condition,x,y)condition成立返回x，否则y

img3 = img2/255
img2val = np.where(img3 >= 0.5, 1, 0)
#print(rgb_point)
#print(img2.shape)
print(height,width)#512X512
print(img)
print("----------")
print(img2)
print("----------")
print(img3)
print("----------")
print(img2val)
