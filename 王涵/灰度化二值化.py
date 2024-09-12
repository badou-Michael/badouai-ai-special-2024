import cv2
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np
#PIL加载图像
#import1.pil=image.open('lenna.png')
#cv2 加载图像
img=cv2.imread('D:/PyCharm/lenna.png')
h,w=img.shape[:2]#是一个切片操作，它从img.shape元组中取出前两个元素，即取图像的高度（h）（行数）（y轴，垂直方向上的像素数）和宽度（w）（列数）（x轴，水平方向上的像素数）
img_gray=np.zeros([h,w],img.dtype)#用于生成给定形状和类型的新数组，数组中的所有元素都初始化为0，指定数组的数据类型为img图片的数组类型
for i in range(h):
    for j in range(w):
        m = img[i,j]
        img_gray[i,j] = int(m[0]*0.11 + m[1]*0.59 + m[2]*0.3)
print(m)
print("-----")
print(img_gray)
print("image show gray: %s"%img_gray)
#cv2.imshow("image show gray",img_gray)

plt.subplot(221)
img=plt.imread('D:/PyCharm/lenna.png')
plt.imshow(img) #rgb图像在轴上显示

print("---image lenna----")

img_gray=rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray') #灰度图在轴上显示

# 二值化
img_binary = np.where(img_gray >= 0.5, 1, 0)
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)
plt.subplot(223)
plt.imshow(img_binary, cmap='gray') #二值图在轴上显示
plt.show() #显示当前已经绘制的所有图像
cv2.waitKey(0)
