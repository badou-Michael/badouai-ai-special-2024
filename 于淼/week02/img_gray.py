
from skimage.color import rgb2gray          #将彩色图像转换为灰度图像
import numpy as np
import matplotlib.pyplot as plt    # 绘图框架 类似于matlab
from PIL import Image   #   PIL  图像处理的一个库（Python Imaging Library）
import cv2          #opencv

# # 灰度化
img = cv2.imread("F:\DeepLearning\Code_test\lenna.png")  #导入图片
# h,w = img.shape[:2]  #图片尺寸   获取图片high和wide
# img_gray = np.zeros([h,w],img.dtype)  # 创建一张和当前图片大小一样的单通道图片
# for i in range(h):
#     for j in range(w):
#         m = img[i,j]  # 获取当前图片的BGR坐标
#         img_gray[i,j] = int(m[0]*0.11+m[1]*0.59+m[2]*0.3)  # 把图片BGR坐标转化成灰度坐标
# print(m)
# print(img_gray)
# print("图片灰度值：%s"%img_gray)
# cv2.imshow("image_gray",img_gray)
#
# plt.subplot(221)                #创建画布   1行3列第1张图片
# img = plt.imread("lenna.png")
# plt.imshow(img)
# print("---image lenna----")
# print(img)

# 灰度化
img_gray = rgb2gray(img)
plt.subplot(222)            #两行两列第二个图
plt.imshow(img_gray, cmap='gray')       #   cmap = 'jet'   是热度图象
print("---image gray----")
print(img_gray)


plt.subplot(224)
plt.imshow(img_gray,cmap='jet')
print("----img jet-----")

# 二值化
# rows, cols = img_gray.shape
# for i in range(rows):
#     for j in range(cols):
#         if (img_gray[i, j] <= 0.5):
#             img_gray[i, j] = 0
#         else:
#             img_gray[i, j] = 1

img_binary = np.where(img_gray >= 0.5, 1, 0)
#1、np.where(condition,x,y) 当where内有三个参数时，第一个参数表示条件，当条件成立时where方法返回x，当条件不成立时where返回y
#2、np.where(condition) 当where内只有一个参数时，那个参数表示条件，当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
print("-----imge_binary------")
print(img_binary)
print(img_binary.shape)

plt.subplot(223)
plt.imshow(img_binary, cmap='gray')
plt.show()      #将创建的图形显示在屏幕上或保存到文件中
