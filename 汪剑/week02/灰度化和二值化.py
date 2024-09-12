from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
import cv2

# #灰度化
# img = cv2.imread('lenna.png')
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) # opencv读图通道排列为BGR，转成RGB
# h,w = img.shape[:2]
#
# #创建一张和当前大小一样的单通道图
# img_gray = np.zeros([h,w],img.dtype)
# for i in range(h):
#     for j in range(w):
#         m = img[i,j]  #获取RGB坐标
#         img_gray[i,j] = int(m[0]*0.3 + m[1]*0.59 + m[2]*0.11)  # RGB转GRAY，浮点算法：GRAY = 0.3R + 0.59G + 0.11B
#
# # cv2.imshow('img_gray：',img_gray)
# # cv2.waitKey(0) # 在主线程中等待键盘事件 0或者空值表示无限等待，直到检测到键盘输入为止
# # cv2.destroyAllWindows() #用于关闭所有由cv2.imshow()打开的窗口。它通常在程序结束时调用，以确保所有窗口都被正常关闭，避免占用系统资源
#
# #二值化
# img_binary = np.where(img_gray/255 >= 0.5, 1, 0)
#
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img_gray,cmap='gray')
# plt.subplot(223)
# plt.imshow(img_binary,cmap='gray')
# plt.show()


#灰度化
img = cv2.imread('lenna.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img_gray= cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
#二值化
_,img_binary = cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY)

plt.subplot(221)
plt.imshow(img)
plt.subplot(222)
plt.imshow(img_gray,cmap='gray')
plt.subplot(223)
plt.imshow(img_binary,cmap='gray')
plt.show()


# #灰度化
# img = plt.imread('lenna.png')
# img_gray = rgb2gray(img)
# print(img_gray)
# #二值化
# img_binary = np.where(img_gray >= 0.5,1,0)
# print(img_binary)
#
# plt.subplot(221)
# plt.imshow(img)
# plt.subplot(222)
# plt.imshow(img_gray,cmap='gray')
# plt.subplot(223)
# plt.imshow(img_binary,cmap='gray')
# plt.show()
