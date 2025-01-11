import cv2
import numpy as np
from matplotlib import pyplot as plt

img_path= r"lenna.png"
img=cv2.imread(img_path)

#灰度直方图  案例 1
#Open CV默认为bGr
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
plt.figure()
plt.hist(img_gray.ravel(),256)#第一个参数是数据数组，这里是 gray.ravel()，表示图像中的所有像素值。 ravel() 来确保数据是一维的
# ## 第二个参数 256 指定了直方图的桶数，确保每个可能的像素值都有一个对应的桶。
# cv2.imshow("lena gray",img_gray)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()
plt.show()

##灰度直方图 案例2 plt.plot() 绘制的直方图看起来像折线图，而 plt.hist() 绘制的则更像传统的条状直方图。
'''
img = cv2.imread("lenna.png", 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##灰度直方图2
hist=cv2.calcHist ([img_gray],[0],None,[256],[0,256])
plt.figure()#新建图
plt.xlabel('bins') #x轴
plt.ylabel('piexl')#y周
plt.plot(hist) #cv2.calcHist 返回的 hist 是一个一维的 NumPy 数组，其中每个元素对应一个灰度级（从0到255）的像素计数。这个数组本身就是一维的，因此不需要使用 ravel() 来展平它。
plt.xlim(0,256)#设置x坐标轴范围
plt.show()
'''
##彩图直方图
'''
img_path= r"lenna.png"
img = cv2.imread("lenna.png", 1)
chans=cv2.split(img)
colors=("blue","green","red")
plt.figure()
plt.title("Flattened Color Histogram")
plt.xlabel("Bins")
plt.ylabel("# of Pixels")

for chan,color in zip(chans,colors):
    hist=cv2.calcHist([chan],[0],None,[256],[0,256])
    plt.plot(hist,color=color)
    plt.xlim([0,256])
plt.show()
'''

################均衡化
import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
equalizeHist—直方图均衡化
函数原型： equalizeHist(src, dst=None)
src：图像矩阵(单通道图像)
dst：默认即可

src：输入图像，必须是单通道的灰度图像。
dst：输出图像，与输入图像具有相同的尺寸和类型。

注意事项
直方图均衡化通常只适用于灰度图像。对于彩色图像，你需要对每个颜色通道分别进行均衡化。
直方图均衡化可能会增强图像中的噪声，特别是在图像的暗部或亮部区域。
扩展到彩色图像
对于彩色图像，你可以将图像分解为单独的颜色通道，对每个通道分别进行直方图均衡化，然后再将它们合并回去
'''
##灰度图像均值化
img=cv2.imread("lenna.png")
img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
##cv2.imshow('gray_lenna',img_gray)
##cv2.waitKey(0)
dst=cv2.equalizeHist(img_gray)
##z灰度后直方图像素计算
hist=cv2.calcHist([img_gray],[0],None,[256],[0,256])
plt.figure()
#plt.plot(hist)
plt.hist(img_gray.ravel(),256)
plt.show()

cv2.imshow('Histogram Equalization',np.hstack([img_gray,dst]))
#np.hstack([gray, dst]) 将两个图像数组 gray 和 dst 水平堆叠在一起。gray 是原始的灰度图像，而 dst 是经过直方图均衡化后的图像
cv2.waitKey(0)





# 彩色图像直方图均衡化
img=cv2.imread('lenna.png',1)
b,g,r=cv2.split(img)##cv2.split 函数用于将多通道图像分割成单独的通道。这个函数将图像的每个颜色通道（例如，彩色图像的红色、绿色和蓝色通道）分割成独立的数组。
bh=cv2.equalizeHist(b)
gh=cv2.equalizeHist(g)
rh=cv2.equalizeHist(r)

img_merge=cv2.merge((bh,gh,rh))
## cv2.merge 函数需要一次处理多个通道图像，而不是单独处理每个通道。
# 元组提供了一种方便的方式来打包这些通道，使得它们可以作为一个整体被传递给函数。
#在使用 cv2.merge 时使用两个括号来创建一个元组，这是正确传递多个通道图像给函数的唯一方式。cv2.merge((bh,gh,rh))
cv2.imshow('Color Image Histogram Equalization"',img_merge)
cv2.waitKey(0)
















