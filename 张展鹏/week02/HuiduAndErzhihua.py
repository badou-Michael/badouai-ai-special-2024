import cv2
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
'''
1.通过读取图片，通过计算将RGB转为Gray实现灰度
2.通过CV2读取图片，通过函数实现灰度
3.通过rgb2gray函数实现灰度
4.通过手动计算灰度值
5.通过plt读取图片，并通过rgb2gray函数实现灰度，判断二值化
'''

imgAdr = plt.imread('lbxx.jpg') #使用plt读取，plt读取是RGB
plt.subplot(231) #将原图片展示在2*3方格中的第一个位置
plt.imshow(imgAdr)


#1.通过手动计算
imgAdr = cv2.imread('lbxx.jpg') #使用cv2读取图片
h,w,n = imgAdr.shape #分别获取图片各项指标大小
imgHd = np.zeros([h,w],imgAdr.dtype) #使用np.zeros创建一个和当前图片一样大小的单通道图片，全黑（数据全为0的h行w列矩阵）
for i in range(h): #因为灰度图只有一个颜色通道，n=1，所以n不计入循环
    for j in range(w): #因为通过CV2读取，每个矩阵中存在三色值，需要通过计算算出对应灰度值
        a = imgAdr[i,j] #取出当前图片的BGR坐标
        imgHd[i,j] = (a[0] * 0.11 + a[1] * 0.59 + a[2] * 0.3) #公式：Gray=0.3R+0.59G+0.11B,因为CV2是读取的BGR，所以计算顺序改变

print("--------------------------原图片数据--------------------------")
print(imgAdr)
print("--------------------------灰度后数据--------------------------")
print(imgHd)
cv2.imshow("show cv2Gray",imgHd)
cv2.waitKey(0)

#2.通过CV2读取，直接通过函数
imgAdr = cv2.imread('lbxx.jpg')
imgHd = cv2.cvtColor(imgAdr,cv2.COLOR_RGB2GRAY) #通过cv2的函数直接将图片转换为RGB灰度后的值
print("--------------------------cv2函数灰度后数据--------------------------")
print(imgHd)
plt.subplot(232)
plt.imshow(imgHd,cmap='gray')

#3.通过rgb2gray函数实现灰度
imgAdr = plt.imread('lbxx.jpg')
imgHd = rgb2gray(imgAdr)
print("--------------------------rgb2gray函数灰度后数据--------------------------")
plt.subplot(233)
plt.imshow(imgHd,cmap='gray')

#4.通过手动计算灰度值-通过cv2读取并没有进行归一化，所以需要通过0~255去比较
imgAdr = cv2.imread('lbxx.jpg')
imgHd = cv2.cvtColor(imgAdr,cv2.COLOR_RGB2GRAY) #通过cv2的函数直接将图片转换为RGB灰度后的值
a,b = imgHd.shape
for i in range(a):
    for j in range(b):
        if(imgHd[i,j] > 128):
            imgHd[i,j] = 1
        else:
            imgHd[i,j] = 0
print("--------------------------手动二值化后数据--------------------------")
print(imgHd)
plt.subplot(234)
plt.imshow(imgHd,cmap='gray')

#5.通过plt读取图片，并通过rgb2gray函数实现灰度，判断二值化
imgAdr = plt.imread('lbxx.jpg')
imgHd = rgb2gray(imgAdr)
imgbry = np.where(imgHd >= 0.5,1,0)
print("--------------------------二值化后数据--------------------------")
print(imgbry)
plt.subplot(235)
plt.imshow(imgbry,cmap='gray')
plt.show()


