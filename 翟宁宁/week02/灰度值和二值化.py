'''
第二周作业 ,实现图像的灰度化,二值化

实现思路：
1.读取原图像
2.拿到所有像素,对每一像素使用 灰度值算法
灰度值算法 ：Gray=R0.3+G0.59+B0.11
3.使用画布输出目标图像（灰度图，二值图）

'''


import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


#1读原图
img = cv.imread("./images/lenna.png")
#print(img)  #img是一个三维数组（h,w,c）,有三个通道（RGB）
#img1 = img.shape[0:]
#print(img1)  #(512, 512, 3)
h,w = img.shape[:2]
img_gray = np.zeros([h,w],img.dtype)   #创建一个由0组成的二维数组 , hxw

#Gray=R0.3+G0.59+B0.11
print(img)
print("---------------------------")
for i in range(h):
    for j in range(w):
        temp = img[i,j]
        # print(temp)
        img_gray[i,j] = int(temp[0]*0.11+temp[1]*0.59+temp[2]*0.3)

print(temp)
print("---------------------------")
print(img_gray)
print("---------------------------")


#在画布显示

plt.subplot(221)
img = plt.imread("./images/lenna.png")
plt.imshow(img)
print(img)

plt.subplot(222)
plt.imshow(img_gray, cmap='gray') #这里一个问题 cmap不写显示为绿色 ，
print("----------------------")
print(img_gray)


#二值化
h,w = img_gray.shape
for i in range(h):
    for j in range(w):
        if (img_gray[i,j]/255) >= 0.5:
            img_gray[i,j] = 0
        elif (img_gray[i,j]/255) < 0.5:
            img_gray[i,j] = 1


plt.subplot(223)
plt.imshow(img_gray)
print("------------------")
print(img_gray)



plt.show()
