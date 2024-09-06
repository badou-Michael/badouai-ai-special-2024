# noinspection PyUnresolvedReferences
import numpy as np
# noinspection PyUnresolvedReferences
import cv2
# noinspection PyUnresolvedReferences
import matplotlib.pyplot as plt
# noinspection PyUnresolvedReferences
from skimage.color import rgb2gray

#灰度化
# 法一 手动算
image=cv2.imread("lenna.png")#读图片
h,w=image.shape[:2]#获取图片的高和宽
image_new=np.zeros((h,w),image.dtype)#新建一张和image高和宽以及数据类型一样的以0填充（全黑）的图片 (h,w)=h行*w列的矩阵
#print(image_new)
for i in range(h):# in 范围
    for j in range(w):
        m=image[i,j]#取出当前high和wide中的BGR坐标 因为是用CV2读取的
        image_new[i,j]=m[2]*0.3+m[1]*0.59+m[0]*0.11#通过列表顺序下标方法取到红、绿、蓝的值;将BGR坐标转化为gray坐标并赋值给新图像
#print(m)
#print(image_new)
#展示图片法一
plt.subplot(221)#展示的图片将要放在整个画布的什么位置。第一个2为行数，第二个2为列数，1为位置
plt.imshow(image_new,cmap='gray')# 展示灰色图像;后面不跟 cmap='',展示彩色图片
plt.show()#显示图像窗口 有了这个图像才能被展示出来

#展示图片法二
#cv2.imshow("show image gray",image_gray)#引号里的为窗口名
#cv2.waitKey(0)

#灰度化
# 法二 直接调函数
#1 用CV
image=cv2.imread("lenna.png")
image_gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("show image gray",image_gray)
cv2.waitKey(0)

#2 用rgb2gray
image=cv2.imread("lenna.png")
image_gray=rgb2gray(image)
plt.subplot(111)
plt.imshow(image_gray,cmap='gray')
plt.show()

#二值化
#法1 手动设
h1,w1=image_new.shape
#image_wb=np.zeros((h,w),image_new.dtype)#新建一张和image高和宽以及数据类型一样的以0填充（全黑）的图片 (h,w)=h行*w列的矩阵
for i1 in range(h1):
    for j1 in range(w1):
        if (image_new[i1, j1] <=128):#因为上面是用CV2读的图片，没有经过归一化，所以没有用0.5作为临界点。如果用matplotlib.pyplot 读取图片后，会自动归一化成0-1，然后就用0.5作为临界点
            image_new[i1, j1]=0
        else:
            image_new[i1, j1] = 255
plt.subplot(111)
plt.imshow(image_new,cmap='gray')
plt.show()

#二值化
#法2
image_wb=np.where(image_new <= 128,0,255)#因为image_new是经过灰度化的，每个像素只有一个值，所以可以和128做比较
plt.subplot(111)
plt.imshow(image_wb,cmap='gray')
plt.show()








