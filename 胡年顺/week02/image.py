
# -*- coding: utf-8 -*-
"""
第二周作业
@author: 胡年顺
"""
from skimage.color import rgb2gray
import  numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# 使用类库cv2读取一张图片，返回一个表示图片的numpy数组
img = cv2.imread("img/lenna.png")
# cv2.imshow("image show gray",img)
# cv2.waitKey(0)

# ChangeGray图片灰度化
def ChangeGray( img ):
    # img.shape返回数组的元组，通常是[height, width, channels],channels为通道
    # h, w, c = img.shape  # 512 512 3
    h,w = img.shape[:2] #获取元组的前两位  h:高度  w:宽度
    # 创建一张大小为hw的相同图片大小的单通道图片
    img_gray=np.zeros([h,w],img.dtype)
    for i in range(h):
        for j in range(w):
            #注意cv2类库取出的像素点坐标为BGR ，非RGB
            m=img[i,j] #取出当前高度和宽度中BGR坐标
            img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)
            # print(m)
    print("image show gray %s"%img_gray)
    cv2.imshow("image show gray",img_gray)  #展示处理完的图片
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#使用自定义函数
# ChangeGray(img)

#使用类库skimage.color函数直接转换
# img_gray = rgb2gray(img)
# print(img_gray)
# cv2.imshow("image show gray",img_gray)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


#当前函数使用plt类库设置了一个2*2表格，最后一个1表示把图片放到第1个位置上
plt.subplot(221)
img = plt.imread("img/lenna.png")
plt.imshow(img)


#灰度图
img_gray = rgb2gray(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')

#二值图1
rows,clos =img_gray.shape
for i in range(rows):
    for j in range(clos):
        if img_gray[i,j] <= 0.5:
            img_gray[i,j] = 0
        else:
            img_gray[i, j] = 1
plt.subplot(223)
plt.imshow(img_gray, cmap='gray')

#二值图2
plt.subplot(224)
#使用np.where函数直接调用
img_binary = np.where(img_gray >= 0.5, 1, 0)
print(img_binary)
print(img_binary.shape)
plt.imshow(img_gray, cmap='gray')
plt.show()
